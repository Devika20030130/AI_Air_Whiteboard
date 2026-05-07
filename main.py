import cv2
import numpy as np
import mediapipe as mp
import pytesseract
import sympy as sp
import os

# ---------------- Tesseract ----------------
pytesseract.pytesseract.tesseract_cmd = os.getenv(
    "TESSERACT_PATH",
    "tesseract"
)

print("🚀 AI Whiteboard v3 Started")

# ---------------- Mediapipe ----------------
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# ---------------- Camera ----------------
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

canvas = None
prev_x, prev_y = 0, 0

# Prevent accidental clear
clear_frames = 0

# ---------------- OCR FUNCTION ----------------
def recognize(canvas_img):

    gray = cv2.cvtColor(
        canvas_img,
        cv2.COLOR_BGR2GRAY
    )

    _, thresh = cv2.threshold(
        gray,
        120,
        255,
        cv2.THRESH_BINARY_INV
    )

    text = pytesseract.image_to_string(thresh)

    return text.strip()

# ---------------- Finger State ----------------
def fingers_up(lm):

    index_up = lm[8].y < lm[6].y
    middle_up = lm[12].y < lm[10].y
    ring_up = lm[16].y < lm[14].y
    pinky_up = lm[20].y < lm[18].y

    return (
        index_up,
        middle_up,
        ring_up,
        pinky_up
    )

# ---------------- MAIN LOOP ----------------
while True:

    ret, frame = cap.read()

    if not ret:
        break

    frame = cv2.flip(frame, 1)

    h, w, _ = frame.shape

    if canvas is None:
        canvas = np.zeros((h, w), dtype=np.uint8)

    rgb = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB
    )

    result = hands.process(rgb)

    # =====================================================
    # HAND DETECTION
    # =====================================================

    if result.multi_hand_landmarks:

        # -------------------------------------------------
        # ✊ CLOSED FIST => CLEAR BOARD
        # -------------------------------------------------

        for hand in result.multi_hand_landmarks:

            lm = hand.landmark

            index_down = lm[8].y > lm[6].y
            middle_down = lm[12].y > lm[10].y
            ring_down = lm[16].y > lm[14].y
            pinky_down = lm[20].y > lm[18].y

            # All fingers folded
            if (
                index_down and
                middle_down and
                ring_down and
                pinky_down
            ):

                clear_frames += 1

                if clear_frames > 15:

                    canvas = np.zeros(
                        (h, w),
                        dtype=np.uint8
                    )

                    clear_frames = 0

                    cv2.putText(
                        frame,
                        "BOARD CLEARED",
                        (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (0, 0, 255),
                        3
                    )

            else:
                clear_frames = 0

        # -------------------------------------------------
        # PROCESS EACH HAND
        # -------------------------------------------------

        for hand in result.multi_hand_landmarks:

            lm = hand.landmark

            mp_draw.draw_landmarks(
                frame,
                hand,
                mp_hands.HAND_CONNECTIONS
            )

            # Index finger tip
            x = int(lm[8].x * w)
            y = int(lm[8].y * h)

            # Palm center
            palm_x = int(lm[9].x * w)
            palm_y = int(lm[9].y * h)

            (
                index_up,
                middle_up,
                ring_up,
                pinky_up
            ) = fingers_up(lm)

            # =====================================================
            # ✍️ DRAW MODE
            # =====================================================

            if index_up and not middle_up:

                if prev_x == 0 and prev_y == 0:
                    prev_x, prev_y = x, y

                cv2.line(
                    canvas,
                    (prev_x, prev_y),
                    (x, y),
                    255,
                    8
                )

                prev_x, prev_y = x, y

                cv2.circle(
                    frame,
                    (x, y),
                    8,
                    (0, 255, 0),
                    -1
                )

            # =====================================================
            # ✋ ERASER MODE (OPEN PALM)
            # =====================================================

            elif (
                index_up and
                middle_up and
                ring_up and
                pinky_up
            ):

                erase_radius = 40

                cv2.circle(
                    canvas,
                    (palm_x, palm_y),
                    erase_radius,
                    0,
                    -1
                )

                cv2.circle(
                    frame,
                    (palm_x, palm_y),
                    erase_radius,
                    (0, 0, 255),
                    3
                )

                cv2.putText(
                    frame,
                    "ERASER",
                    (50, 150),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 0, 255),
                    2
                )

                prev_x, prev_y = 0, 0

            # =====================================================
            # ✌️ SPACE MODE
            # =====================================================

            elif index_up and middle_up:

                prev_x, prev_y = 0, 0

                cv2.putText(
                    frame,
                    "SPACE",
                    (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (255, 255, 0),
                    2
                )

            # =====================================================
            # DEFAULT
            # =====================================================

            else:
                prev_x, prev_y = 0, 0

    # =====================================================
    # OVERLAY
    # =====================================================

    output = cv2.add(
        frame,
        cv2.cvtColor(
            canvas,
            cv2.COLOR_GRAY2BGR
        )
    )

    cv2.imshow(
        "AI Whiteboard v3",
        output
    )

    key = cv2.waitKey(1)

    # =====================================================
    # OCR + SOLVE
    # =====================================================

    if key == ord('r'):

        cv2.imwrite(
            "drawing.png",
            canvas
        )

        img = cv2.imread("drawing.png")

        text = recognize(img)

        print("\n🧠 Recognized:", text)

        try:

            result_expr = sp.sympify(text)

            print("➗ Result:", result_expr)

        except:

            print("⚠ Not a valid equation")

    # =====================================================
    # QUIT
    # =====================================================

    elif key == ord('q'):
        break

# =====================================================
# CLEANUP
# =====================================================

cap.release()
cv2.destroyAllWindows()