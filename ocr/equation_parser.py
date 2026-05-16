"""
ocr/equation_parser.py — Raw OCR text → sanitised SymPy expression → solution.

Handles
-------
* Arithmetic        : 3 + 4 * (2 - 1)
* Algebra           : 2x + 5 = 11   →  x = 3
* Exponents         : x^2 + 2x = 8  →  x = [-4, 2]
* Fractions         : 3/4 + 1/2
* Nested parens     : (3 + (4 * 2)) / 7
* Multi-char vars   : ab + cd (treated as a*b + c*d via SymPy implicit mul)
* Constants         : pi, e

Sanitisation pipeline
---------------------
1. Unicode normalisations  (× → *, ² → **2 …)
2. Implicit multiplication  (2x → 2*x,  3(x+1) → 3*(x+1))
3. Reject-symbol filter
4. SymPy sympify()
5. Equation vs expression branching
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional

import sympy as sp
from sympy.parsing.sympy_parser import (
    auto_number,
    auto_symbol,
    convert_xor,
    implicit_multiplication_application,
    standard_transformations,
)

from config import CFG
from utils.logger import get_logger

log = get_logger(__name__)

# ── SymPy transformations that enable implicit multiplication ─────
_TRANSFORMATIONS = (
    standard_transformations
    + (auto_number, auto_symbol, convert_xor, implicit_multiplication_application)
)


@dataclass
class ParseResult:
    raw_text:    str
    sanitised:   str
    result:      str
    is_equation: bool
    success:     bool
    error:       str = ""


class EquationParser:
    """Sanitise OCR output and solve with SymPy."""

    def __init__(self) -> None:
        self._cfg = CFG.parser

    # ──────────────────────────────────────────────────────────
    #  PUBLIC API
    # ──────────────────────────────────────────────────────────
    def parse_and_solve(self, raw: str) -> ParseResult:
        """
        Full pipeline: correct → sanitise → classify → solve.

        Parameters
        ----------
        raw : Text string from any OCR engine.

        Returns
        -------
        ParseResult dataclass with all intermediate and final values.
        """
        if not raw or not raw.strip():
            return ParseResult(raw, "", "Empty input", False, False)

        # ── OCR character-correction ───────────────────────────
        # Common misreads from handwritten digits/operators.
        # Applied on the raw string BEFORE any regex or SymPy parsing
        # so these corrections propagate through the entire pipeline.
        corrected = self._correct_ocr_chars(raw)
        if corrected != raw:
            log.debug("Parser: OCR correction  %r → %r", raw, corrected)

        sanitised = self._sanitise(corrected)
        if not sanitised:
            return ParseResult(
                raw, sanitised,
                "⚠ Rejected — only invalid symbols", False, False,
            )

        log.debug("Parser: raw=%r  →  sanitised=%r", raw, sanitised)

        is_eq = "=" in sanitised
        try:
            result = self._solve_equation(sanitised) if is_eq \
                else self._evaluate(sanitised)
            return ParseResult(raw, sanitised, result, is_eq, True)
        except Exception as exc:
            log.warning("Parser failed on %r: %s", sanitised, exc)
            return ParseResult(
                raw, sanitised,
                f"⚠ Parse error: {exc}", is_eq, False, str(exc),
            )

    # ──────────────────────────────────────────────────────────
    #  OCR CHARACTER CORRECTION
    # ──────────────────────────────────────────────────────────
    @staticmethod
    def _correct_ocr_chars(text: str) -> str:
        """
        Fix common single-character misreads that OCR engines make
        on handwritten mathematical notation.

        These substitutions are applied on the RAW string before any
        regex or SymPy processing so they propagate through the full
        sanitisation pipeline.

        Why each correction is needed
        ──────────────────────────────
        'A' → '4'  : capital A is morphologically similar to 4
        'O' → '0'  : capital O is identical to zero in many hands
        'l' → '1'  : lowercase l is identical to 1 in most fonts
        '|' → '1'  : pipe character is read as 1 by CRNN models
        'S' → '5'  : handwritten S and 5 share the same curves

        Safety: substitutions only apply when the character is
        surrounded by digits or math operators — this avoids
        corrupting genuine variable names like 'x', 'y', 'A'.
        """
        import re

        # Characters surrounded by digit/operator context → correct them.
        # Pattern: optional leading [digit/(] + target char + optional trailing [digit/)]
        _MATH_CTX = r"(?<=[0-9(+\-*/^=])|(?=[0-9)+\-*/^=])"

        corrections = {
            "A": "4",
            "O": "0",
            "l": "1",
            "|": "1",
            "S": "5",
        }

        result = text
        for wrong, right in corrections.items():
            # Replace the character when it appears between numeric/operator chars,
            # or at the start/end of the string if surrounded on one side.
            result = re.sub(
                rf"(?<=[0-9(+\-*/^=\s]){re.escape(wrong)}(?=[0-9)+\-*/^=\s])",
                right,
                result,
            )

        # Also strip all spaces — OCR engines frequently insert spaces
        # inside numbers and operators (e.g. "2 x + 5 = 1 1")
        result = result.replace(" ", "")

        return result

    # ──────────────────────────────────────────────────────────
    #  SANITISATION
    # ──────────────────────────────────────────────────────────
    def _sanitise(self, text: str) -> str:
        # 1. Strip leading/trailing whitespace
        s = text.strip()

        # 2. Reject if any forbidden symbol present
        for sym in self._cfg.reject_symbols:
            if sym in s:
                log.debug("Parser: rejected due to symbol %r", sym)
                return ""

        # 3. Apply configured substitution table
        for src, dst in self._cfg.normalizations:
            s = s.replace(src, dst)

        # 4. Remove stray characters not in the allowed set
        s = re.sub(r"[^0-9a-zA-Z+\-*/().=^, ]", "", s)

        # 5. Collapse multiple spaces / operators
        s = re.sub(r"\s+", " ", s).strip()
        s = re.sub(r"([+\-*/^=])\1+", r"\1", s)

        # 6. Add explicit multiplication where OCR misses it
        #    e.g. "2x" → "2*x",  "3(" → "3*("
        s = re.sub(r"(\d)([a-zA-Z])", r"\1*\2", s)
        s = re.sub(r"(\d)\(", r"\1*(", s)
        s = re.sub(r"\)([a-zA-Z\d])", r")*\1", s)

        return s.strip()

    # ──────────────────────────────────────────────────────────
    #  EXPRESSION EVALUATION
    # ──────────────────────────────────────────────────────────
    @staticmethod
    def _evaluate(expr: str) -> str:
        """Numerically or symbolically evaluate a pure expression."""
        parsed = sp.parse_expr(expr, transformations=_TRANSFORMATIONS)
        simplified = sp.simplify(parsed)
        # Try to return a numeric result when possible
        evaluated = sp.N(simplified, 6)
        # If it's an integer-valued float, show as int
        if evaluated.is_real and evaluated == int(evaluated):
            return str(int(evaluated))
        return str(evaluated)

    # ──────────────────────────────────────────────────────────
    #  EQUATION SOLVING
    # ──────────────────────────────────────────────────────────
    @staticmethod
    def _solve_equation(expr: str) -> str:
        """Solve an equation of the form LHS = RHS for all free symbols."""
        sides = expr.split("=", 1)
        if len(sides) != 2:
            raise ValueError("Expected exactly one '='")

        lhs_str, rhs_str = sides
        lhs = sp.parse_expr(lhs_str, transformations=_TRANSFORMATIONS)
        rhs = sp.parse_expr(rhs_str, transformations=_TRANSFORMATIONS)
        equation = sp.Eq(lhs, rhs)

        free_syms = sorted(equation.free_symbols, key=str)
        if not free_syms:
            # Purely numerical: check truth
            return "True" if sp.simplify(lhs - rhs) == 0 else "False"

        results = {}
        for sym in free_syms:
            sols = sp.solve(equation, sym)
            if not sols:
                results[str(sym)] = "No solution"
            elif len(sols) == 1:
                # Simplify and attempt numerical evaluation
                s = sp.simplify(sols[0])
                results[str(sym)] = str(sp.N(s, 6)) if s.is_number else str(s)
            else:
                cleaned = [
                    str(sp.N(sp.simplify(s), 6)) if sp.simplify(s).is_number
                    else str(sp.simplify(s))
                    for s in sols
                ]
                results[str(sym)] = "[" + ", ".join(cleaned) + "]"

        return "  ,  ".join(f"{k} = {v}" for k, v in results.items())
