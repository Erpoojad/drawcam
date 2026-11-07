"""Turns drawing text (CAD text or OCR output) into structured callouts.

Handles the conventions on typical Indian/ISO job-shop drawings:
  Ø25  ⌀25 h7  R5  M20x1.5  M10  4X Ø8 THRU  Ø6.8 DP 15  M8 DP 12
  25±0.1  25 +0.02/-0.01  C1  1x45°  THK 20  MATERIAL: EN8  SCALE 1:2
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional

COARSE_PITCH = {3: 0.5, 4: 0.7, 5: 0.8, 6: 1.0, 8: 1.25, 10: 1.5, 12: 1.75, 14: 2.0, 16: 2.0,
                18: 2.5, 20: 2.5, 22: 2.5, 24: 3.0, 27: 3.0, 30: 3.5, 33: 3.5, 36: 4.0, 42: 4.5, 48: 5.0}

MATERIALS = {
    "EN1A": "MS", "EN3": "MS", "MS": "MS", "MILD STEEL": "MS", "IS2062": "MS", "C20": "MS",
    "EN8": "EN8", "C45": "EN8", "EN9": "EN8", "EN19": "EN24", "EN24": "EN24", "42CRMO4": "EN24",
    "SS304": "SS304", "SS 304": "SS304", "SS316": "SS304", "SS 316": "SS304", "STAINLESS": "SS304",
    "AL6061": "AL", "AL 6061": "AL", "6061": "AL", "6082": "AL", "ALUMINIUM": "AL", "ALUMINUM": "AL",
    "BRASS": "BRASS", "CI": "CI", "CAST IRON": "CI", "FG260": "CI", "GG25": "CI",
}

NUM = r"(\d+(?:[.,]\d+)?)"

# ISO 286 (microns) for common fits, size ranges up to 180 mm
_R = [3, 6, 10, 18, 30, 50, 80, 120, 180]
_IT = {6: [6, 8, 9, 11, 13, 16, 19, 22, 25], 7: [10, 12, 15, 18, 21, 25, 30, 35, 40], 8: [14, 18, 22, 27, 33, 39, 46, 54, 63]}
_ES = {"g": [-2, -4, -5, -6, -7, -9, -10, -12, -14], "f": [-6, -10, -13, -16, -20, -25, -30, -36, -43], "h": [0] * 9}
_EI = {"k": [0, 1, 1, 1, 2, 2, 2, 3, 3], "m": [2, 4, 6, 7, 8, 9, 11, 13, 15], "p": [6, 12, 15, 18, 22, 26, 32, 37, 43]}


def fit_limits(dia: float, fit: str):
    """(upper, lower) deviation in mm for e.g. H7, h6, g6, f7, k6, m6, p6; None if unknown."""
    try:
        letter, grade = fit[0], int(fit[1:])
        i = next(k for k, r in enumerate(_R) if dia <= r)
        it = _IT[grade][i]
    except (StopIteration, KeyError, ValueError, IndexError):
        return None
    if letter == "H":
        return (it / 1000, 0.0)
    if letter in _ES:
        es = _ES[letter][i]
        return (es / 1000, (es - it) / 1000)
    if letter in _EI:
        ei = _EI[letter][i]
        return ((ei + it) / 1000, ei / 1000)
    return None


def normalize(raw: str, ocr: bool = False) -> str:
    s = raw.strip()
    for a in ("%%C", "%%c", "⌀", "∅", "ø", "Φ", "φ", "\\U+2205", "\\U+00D8"):
        s = s.replace(a, "Ø")
    s = s.replace("%%P", "±").replace("%%p", "±").replace("%%D", "°").replace("%%d", "°")
    s = s.replace("×", "x").replace("X ", "x ").replace(",", ".")
    s = re.sub(r"\s+", " ", s)
    if ocr:
        # Tesseract commonly reads Ø as @ © g o O 0 or Q in front of a number.
        s = re.sub(r"(^|\s)[@©gQoO](?=\d)", r"\1Ø", s)
        s = re.sub(r"(^|\s)0(?=\d{1,3}(?:\.\d+)?(?:\s|$|[hHgGfF]))", r"\1Ø", s)
        s = s.replace("+-", "±").replace("+/-", "±")
        s = re.sub(r"\bC[lI|](?=\b)", "C1", s)                 # C1 chamfer read as Cl
        s = re.sub(r"(?<=\d)[lI|](?=\d|\b)", "1", s)
    return s.upper().replace("Ø", "Ø")


@dataclass
class Callout:
    raw: str
    x: float
    y: float
    conf: float = 1.0
    count: int = 1
    dia: Optional[float] = None
    radius: Optional[float] = None
    length: Optional[float] = None          # bare number = linear dimension
    thread: Optional[tuple] = None          # (major, pitch)
    depth: Optional[float] = None
    through: bool = False
    ream: bool = False
    tol: Optional[tuple] = None             # (upper, lower) deviation
    fit: Optional[str] = None               # h7 / H7 ...
    chamfer: Optional[float] = None
    keywords: set = field(default_factory=set)

    def limits(self, nominal: float):
        if self.tol:
            return self.tol
        if self.fit:
            return fit_limits(nominal, self.fit)
        return None

    def mid_tol_value(self, nominal: float) -> float:
        """Program to the middle of the tolerance band, not the nominal."""
        lim = self.limits(nominal)
        if lim:
            return round(nominal + (lim[0] + lim[1]) / 2, 4)
        return nominal


def parse(raw: str, x=0.0, y=0.0, conf=1.0, ocr=False) -> Callout:
    s = normalize(raw, ocr)
    c = Callout(raw=raw, x=x, y=y, conf=conf)

    m = re.search(r"(\d+)\s*[xX]\s*(?=[ØMR])", s) or re.search(r"(\d+)\s*(?:HOLES|NOS|PLCS)", s)
    if m:
        c.count = int(m.group(1))
    m = re.search(r"\bM\s?" + NUM + r"(?:\s*[xX]\s*" + NUM + ")?", s)
    if m:
        major = float(m.group(1))
        pitch = float(m.group(2)) if m.group(2) else COARSE_PITCH.get(int(round(major)), round(major * 0.15, 2))
        c.thread = (major, pitch)
    m = re.search(r"Ø\s*" + NUM, s)
    if m:
        c.dia = float(m.group(1))
    m = re.search(r"(?:^|\s|X)R\s*" + NUM, s)
    if m and not c.thread:
        c.radius = float(m.group(1))
    m = re.search(r"(?:DP|DEPTH|DEEP|↧|DPT)\s*\.?\s*" + NUM, s) or re.search(NUM + r"\s*(?:DP|DEEP)\b", s)
    if m:
        c.depth = float(m.group(1))
    c.through = bool(re.search(r"\bTHRU\b|\bTHROUGH\b|\bTHR\b", s))
    c.ream = bool(re.search(r"\bREAM|\bH7\b", s)) and c.dia is not None and not c.thread
    m = re.search(r"±\s*" + NUM, s)
    if m:
        v = float(m.group(1))
        c.tol = (v, -v)
    m = re.search(r"\+\s*" + NUM + r"\s*/?\s*-\s*" + NUM, s)
    if m:
        c.tol = (float(m.group(1)), -float(m.group(2)))
    m = re.search(r"\b([A-HJ-NP-Za-hj-np-z])(\d{1,2})\b", raw.replace(" ", " "))
    if m and m.group(1).lower() in "abcdefghjkmnprstuvxyz" and m.group(1).upper() not in ("M", "R", "X", "C"):
        c.fit = m.group(1) + m.group(2)
    m = re.search(r"\bC\s?" + NUM + r"\b", s) or re.search(NUM + r"\s*[xX]\s*45", s)
    if m:
        c.chamfer = float(m.group(1))
    for kw in ("POCKET", "SLOT", "DRILL", "TAP", "CBORE", "C'BORE", "SPOTFACE", "KEYWAY", "GROOVE", "U/C", "UNDERCUT", "THK", "THICK"):
        if kw in s:
            c.keywords.add(kw)
    if c.dia is None and c.thread is None and c.radius is None and c.chamfer is None:
        m = re.fullmatch(r"\(?" + NUM + r"\)?(?:\s*[±+].*)?", s)
        if m:
            c.length = float(m.group(1))
    return c


def parse_globals(texts) -> dict:
    """Title-block style information: material, scale, thickness, part name, qty."""
    out = {"material": None, "scale": None, "thickness": None, "part_name": None, "qty": None}
    joined = " | ".join(normalize(t.text) for t in texts)
    for key in sorted(MATERIALS, key=len, reverse=True):
        if re.search(r"(?<![A-Z0-9])" + re.escape(key) + r"(?![A-Z0-9])", joined):
            out["material"] = MATERIALS[key]
            break
    m = re.search(r"SCALE\s*:?\s*(\d+(?:\.\d+)?)\s*:\s*(\d+(?:\.\d+)?)", joined)
    if m:
        out["scale"] = float(m.group(1)) / float(m.group(2))
    m = re.search(r"(?:THK|THICK(?:NESS)?|T\s*=)\s*:?\s*" + NUM, joined)
    if m:
        out["thickness"] = float(m.group(1))
    m = re.search(r"(?:PART\s*(?:NAME|NO\.?)|DRG\.?\s*NO\.?|TITLE)\s*:?\s*([A-Z0-9\- _/]{2,40}?)(?:\||$)", joined)
    if m:
        out["part_name"] = m.group(1).strip()
    m = re.search(r"QTY\s*:?\s*(\d+)", joined)
    if m:
        out["qty"] = int(m.group(1))
    return out
