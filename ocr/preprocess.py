"""
ocr/preprocess.py — Multi-stage OpenCV preprocessing pipeline
optimised for handwritten mathematical equations on a dark canvas.

Pipeline stages
---------------
1.  Crop to ink bounding box
        Removes empty space — smaller image, faster downstream ops.

2.  Grayscale
        Collapse 3-channel BGR to single-channel intensity.

3.  CLAHE  (Contrast Limited Adaptive Histogram Equalisation)
        Locally equalises brightness so dim operator strokes ( / - + = )
        reach the same contrast level as brighter digit strokes before
        the threshold decision is made.

4.  Upscale x4  (bicubic)
        ~400 DPI equivalent for Tesseract / TrOCR.

5.  Gaussian blur 5x5
        Removes high-frequency ringing from the upscale step.

6.  Adaptive Gaussian threshold  (block=31, C=12)
        Converts grey levels to binary ink — local so uneven lighting
        does not cause half the strokes to disappear.

7.  Dilation  (2x2 rect, 1 iteration)
        Thickens every ink pixel by 1 px in each direction.
        Core fix for operator disappearance: a 2-px '-' bar becomes
        4 px and survives both denoise and contour filtering.
        2x2 chosen deliberately — 3x3 merges adjacent digit strokes.

7b. Fine morphological close  (2x2, 1 iter)
        Closes hairline breaks on anti-aliased edges AFTER dilation.

8.  Morphological close  (3x3 ellipse, 2 iter)
        Bridges larger mid-stroke gaps isotropically.

9.  Contour noise removal
        Drops connected components whose area < min_contour_area.

10. fastNlMeans denoise
        Final speckle suppression.

11. Border padding
        Adds content_padding pixels of black on every side so glyphs
        are not flush against the image edge.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

import cv2
import numpy as np

from config import CFG
from utils.image_utils import (
    BGR,
    BINARY,
    GRAY,
    crop_to_ink,
    make_debug_grid,
    scale_image,
)
from utils.logger import get_logger

log = get_logger(__name__)


# ==================================================================
#  PREPROCESSOR
# ==================================================================
class EquationPreprocessor:
    """
    Converts a raw BGR whiteboard canvas into a clean binary image
    ready for OCR.

    Stateless — every call to process() is independent.
    All kernels are pre-allocated in __init__ and reused across calls.
    """

    def __init__(self) -> None:
        self._cfg = CFG.preprocess

        # ── CLAHE instance (created once, reused) ------------------
        # createCLAHE() is ~1 ms; sharing one instance avoids repeated
        # setup cost on every OCR trigger.
        self._clahe = cv2.createCLAHE(
            clipLimit=self._cfg.clahe_clip_limit,
            tileGridSize=self._cfg.clahe_tile_grid,
        )

        # ── Dilation kernel — 2x2 RECT -----------------------------
        # MORPH_RECT (not MORPH_ELLIPSE) for dilation:
        #   A rect expands equally in X and Y — correct for horizontal
        #   operators like '-' and '='.
        #   An ellipse rounds corners, potentially shrinking the height
        #   of thin horizontals by rounding them away.
        ks = self._cfg.dilation_kernel_size
        self._dilation_kernel: np.ndarray = cv2.getStructuringElement(
            cv2.MORPH_RECT, (ks, ks)
        )

        # ── Fine close kernel — 2x2 rect ---------------------------
        self._fine_kernel: np.ndarray = np.ones((2, 2), np.uint8)

        # ── Main close kernel — 3x3 ellipse ------------------------
        # Ellipse for isotropic gap-filling on rounded stroke ends.
        self._morph_kernel: np.ndarray = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE, (3, 3)
        )

        self.debug_stages: List[Tuple[str, np.ndarray]] = []

    # --------------------------------------------------------------
    #  PUBLIC API
    # --------------------------------------------------------------
    def process(
        self,
        canvas_bgr: BGR,
        collect_debug: bool = False,
    ) -> Optional[BINARY]:
        """
        Run the full 11-stage preprocessing pipeline.

        Parameters
        ----------
        canvas_bgr    : Raw BGR canvas from CanvasManager.
        collect_debug : When True every stage image is appended to
                        self.debug_stages for the debug-grid window.

        Returns
        -------
        Binary uint8 image (ink=255, background=0) ready for OCR,
        or None if the canvas contains no ink.
        """
        self.debug_stages = []

        def _save(label: str, img: np.ndarray) -> None:
            if collect_debug:
                self.debug_stages.append((label, img.copy()))

        _save("0_raw", canvas_bgr)

        # ── Stage 1: Crop to ink -----------------------------------
        # Tight bounding box around all non-zero pixels + padding.
        # Eliminates dead black background so every downstream op
        # works on only the relevant region.
        cropped = crop_to_ink(canvas_bgr, padding=self._cfg.content_padding)
        if cropped is None:
            log.debug("Preprocess: canvas is blank — skipping OCR.")
            return None
        _save("1_crop", cropped)

        # ── Stage 2: Grayscale ------------------------------------
        gray: GRAY = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
        _save("2_gray", gray)

        # ── Stage 3: CLAHE ----------------------------------------
        #
        # Why CLAHE *before* upscale?
        #   CLAHE divides the image into tileGridSize tiles and runs
        #   histogram equalisation locally within each tile.  At the
        #   original resolution each 8x8 tile covers a meaningful
        #   spatial neighbourhood (~1 digit wide).  After 4x upscale
        #   the same tile would cover 1/16 of the area — too small to
        #   capture per-stroke luminance variation and too large to
        #   benefit from local equalisation simultaneously.
        #
        # Effect on thin operators:
        #   Air-drawn '-' and '/' are typically 20-40 % dimmer than
        #   surrounding digit strokes because the finger moves faster
        #   through them.  Adaptive threshold computes a local mean
        #   that includes those dim values; without CLAHE the '-' may
        #   sit below the mean + C threshold and become background.
        #   CLAHE stretches its contrast to match the digits, so the
        #   threshold sees it at full brightness.
        clahe_out: GRAY = self._clahe.apply(gray)
        _save("3_clahe", clahe_out)

        # ── Stage 4: Upscale x4 -----------------------------------
        # Bicubic interpolation preserves stroke edge sharpness better
        # than bilinear (smoother) or nearest-neighbour (blocky).
        scaled: GRAY = scale_image(
            clahe_out,
            self._cfg.ocr_scale,
            interpolation=cv2.INTER_CUBIC,
        )
        _save("4_upscale", scaled)

        # ── Stage 5: Gaussian blur 5x5 ----------------------------
        # Removes bicubic ringing artifacts along stroke edges.
        # Applied in greyscale BEFORE threshold so it softens the
        # edge gradient (benefiting threshold decisions) rather than
        # blurring already-binary ink pixels.
        blurred: GRAY = cv2.GaussianBlur(scaled, (5, 5), 0)
        _save("5_blur", blurred)

        # ── Stage 6: Adaptive threshold ---------------------------
        # THRESH_BINARY_INV: ink pixels -> 255, background -> 0.
        #
        # blockSize=31: neighbourhood spans ~one digit at 4x scale,
        #   wide enough to tolerate local brightness variation while
        #   remaining local enough to handle camera vignetting.
        #
        # C=12: after computing the Gaussian-weighted local mean,
        #   subtract 12 before comparing.  This biases the decision
        #   toward keeping pixels rather than dropping them, which
        #   favours thin operator strokes at the cost of a slightly
        #   noisier background (cleaned by later stages).
        thresh: BINARY = cv2.adaptiveThreshold(
            blurred, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            blockSize=self._cfg.adaptive_block,
            C=self._cfg.adaptive_c,
        )
        _save("6_threshold", thresh)

        # ── Stage 7: Dilation ------------------------------------
        #
        # The primary fix for operator symbol disappearance.
        #
        # Problem: after adaptive threshold at 4x scale, thin
        # operators have the following heights in binary pixels:
        #
        #   Symbol   brush_px   after_4x   binary_height   risk
        #   ------   --------   --------   -------------   ----
        #   '-'         4          16          ~2 px        HIGH
        #   '+'         4          16          ~4 px cross  MED
        #   '='         4          16          ~2 px x2     HIGH
        #   '/'         4          16          ~2 px diag   HIGH
        #
        # A 2-px tall binary bar:
        #   - is erased by fastNlMeans (h=12 removes features < 12 GL)
        #   - has area ~2 * stroke_len px < min_contour_area=80 for
        #     strokes shorter than 40 px — likely for '-' in fractions.
        #
        # Dilation with a 2x2 rect expands each white pixel by 1 px
        # in +X, -X, +Y, -Y simultaneously:
        #   '-' 2px tall  ->  4px tall      (safe from both threats)
        #   '/' 2px wide  ->  4px wide      (safe from both threats)
        #   '+' cross     ->  filled cross  (no interior void)
        #   '=' bars      ->  thicker bars  (both bars individually safe)
        #
        # Why 2x2 and not 3x3?
        #   At 4x scale, the inter-symbol gap for adjacent digits like
        #   '12' is typically 8-12 px.  A 3x3 dilation expands each
        #   side by 1.5 px (rounds to 2), consuming 4 px of a 12 px
        #   gap — 33 % — and risks merging '1' and '2' into one
        #   connected component.  A 2x2 kernel consumes 2 px (17 %),
        #   leaving the gap intact.
        dilated: BINARY = cv2.dilate(
            thresh,
            self._dilation_kernel,
            iterations=self._cfg.dilation_iterations,
        )
        _save("7_dilate", dilated)

        # ── Stage 7b: Fine morphological close --------------------
        # Closes hairline breaks on stroke edges that were introduced
        # by the Gaussian blur + threshold interaction on anti-aliased
        # brush-stroke edges.
        # Runs AFTER dilation: thin operators are already thickened
        # so this pass acts only on real edge-breaks, not on operators.
        fine_closed: BINARY = cv2.morphologyEx(
            dilated, cv2.MORPH_CLOSE, self._fine_kernel
        )
        _save("7b_fine_close", fine_closed)

        # ── Stage 8: Morphological close (ellipse) ----------------
        # Bridges larger gaps (3-8 px) caused by:
        #   - finger speed variation within a single stroke
        #   - brief pen-lift false-positives in the gesture debouncer
        # Ellipse kernel chosen for isotropic (direction-independent)
        # fill — equally effective on vertical, horizontal, diagonal.
        closed: BINARY = cv2.morphologyEx(
            fine_closed,
            cv2.MORPH_CLOSE,
            self._morph_kernel,
            iterations=self._cfg.morph_iterations,
        )
        _save("8_morphclose", closed)

        # ── Stage 9: Contour noise removal ------------------------
        # At 4x scale with prior dilation, minimum operator areas:
        #   '-' (length=20px drawn): 4 * 20 * 16 = 1280 px2  (dilated)
        #   '.' decimal point:       4 * 4  * 16 = 256 px2
        # Camera noise / dust specks: < 80 px2
        # The 80 px2 threshold therefore drops only genuine noise.
        cleaned = self._remove_noise_contours(closed)
        _save("9_contour_clean", cleaned)

        # ── Stage 10: fastNlMeans denoise -------------------------
        # h=12: removes grey-level features smaller than 12 intensity
        # units.  After dilation all real ink is solid 255 (not 12
        # intensity units wide) so denoise only touches residual noise.
        denoised: BINARY = cv2.fastNlMeansDenoising(
            cleaned, h=self._cfg.denoise_h
        )
        _save("10_denoise", denoised)

        # ── Stage 11: Border padding ------------------------------
        # Tesseract and TrOCR both expect context pixels around glyphs.
        # Without padding, a digit at the image edge loses its upper/
        # lower serif to the convolutional front-end's border handling.
        pad = self._cfg.content_padding
        final: BINARY = cv2.copyMakeBorder(
            denoised, pad, pad, pad, pad,
            cv2.BORDER_CONSTANT, value=0,
        )
        _save("11_final", final)

        log.debug(
            "Preprocess: %dx%d → %dx%d  (%d debug stages)",
            canvas_bgr.shape[1], canvas_bgr.shape[0],
            final.shape[1],      final.shape[0],
            len(self.debug_stages),
        )
        return final

    # --------------------------------------------------------------
    #  PROCESS FOR TRANSFORMER  (RGB PIL image)
    # --------------------------------------------------------------
    def process_for_transformer(self, canvas_bgr: BGR):
        """
        Return a PIL RGB image optimised for TrOCR (ViT encoder).

        Lighter pipeline than process() because TrOCR's ViT encoder
        operates on fixed 384x384 patches.  Full 4x upscale creates
        images whose patches span < 1 glyph — counterproductive.
        Upscale capped at 2x.

        CLAHE and dilation are retained because their benefits apply
        regardless of the downstream model architecture.
        """
        cropped = crop_to_ink(canvas_bgr, padding=self._cfg.content_padding)
        if cropped is None:
            return None

        # Grayscale
        gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

        # CLAHE — normalise dim operator strokes before thresholding
        clahe_out = self._clahe.apply(gray)

        # Upscale x2 (max safe for ViT patch grid)
        scaled = scale_image(
            clahe_out,
            min(self._cfg.ocr_scale, 2.0),
            interpolation=cv2.INTER_CUBIC,
        )

        # Otsu threshold -> binary, ink=255 background=0
        _, binary = cv2.threshold(
            scaled, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU,
        )

        # Dilation — same 2x2 rect, 1 iteration
        # Ensures operators are visible to ViT patch extractor
        dilated = cv2.dilate(
            binary,
            self._dilation_kernel,
            iterations=self._cfg.dilation_iterations,
        )

        # Invert: TrOCR expects white background, black ink
        inverted = cv2.bitwise_not(dilated)

        # 3-channel RGB PIL (TrOCRProcessor input format)
        rgb = cv2.cvtColor(inverted, cv2.COLOR_GRAY2RGB)
        from PIL import Image
        return Image.fromarray(rgb)

    # --------------------------------------------------------------
    #  HELPERS
    # --------------------------------------------------------------
    def _remove_noise_contours(self, binary: BINARY) -> BINARY:
        """
        Erase all connected components whose pixel area is below
        min_contour_area.

        After dilation the shortest plausible operator stroke:
            brush_thickness=4 * length=5px * ocr_scale^2=16 = 320 px2
        safely exceeds the 80 px2 threshold, so no real ink is lost.

        RETR_EXTERNAL: only outermost contours are returned, so the
        interior of '0', '8', '6' etc. is not treated as a separate
        component and incorrectly dropped.
        """
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        result  = binary.copy()
        removed = 0
        for cnt in contours:
            if cv2.contourArea(cnt) < self._cfg.min_contour_area:
                cv2.drawContours(result, [cnt], -1, 0, -1)
                removed += 1
        if removed:
            log.debug(
                "Contour filter: dropped %d noise component(s) "
                "(< %d px2)",
                removed, self._cfg.min_contour_area,
            )
        return result

    # --------------------------------------------------------------
    #  DEBUG GRID
    # --------------------------------------------------------------
    def show_debug_window(self, win_name: str = "Preprocess Debug") -> None:
        """
        Display a side-by-side grid of all pipeline stage images.
        Must be called after process(collect_debug=True).
        Caller is responsible for cv2.waitKey().
        """
        if not self.debug_stages:
            log.warning(
                "No debug stages — call process(collect_debug=True) first."
            )
            return
        grid = make_debug_grid(self.debug_stages, target_w=260)
        cv2.imshow(win_name, grid)
