"""PDF reader. Vector PDFs (exported from CAD) give real lines and text;
image-only PDFs (scans) are rasterised and passed to the scan reader."""
from __future__ import annotations

import math
import re

import pymupdf

from .drawing import Drawing, Seg, Circle, Text

PT_TO_MM = 25.4 / 72.0
TITLE_KEYS = re.compile(r"MATERIAL|SCALE|DRG|DRAWN|QTY|TITLE|CHECKED|PART\s*NO|TOLERANCE|FINISH|DATE", re.I)


def _bezier(p0, p1, p2, p3, n=12):
    out = []
    for i in range(n + 1):
        t = i / n
        a = (1 - t) ** 3
        b = 3 * (1 - t) ** 2 * t
        c = 3 * (1 - t) * t ** 2
        d = t ** 3
        out.append((a * p0.x + b * p1.x + c * p2.x + d * p3.x, a * p0.y + b * p1.y + c * p2.y + d * p3.y))
    return out


def _dash_kind(dashes: str) -> str:
    nums = re.findall(r"[\d.]+", (dashes or "").split("]")[0])
    if len(nums) >= 4:
        return "center"
    if len(nums) >= 2:
        return "hidden"
    return "visible"


def is_vector(path: str) -> bool:
    doc = pymupdf.open(path)
    page = doc[0]
    return len(page.get_drawings()) > 10 or len(page.get_text("words")) > 5


def read_pdf(path: str, page_no: int = 0) -> Drawing:
    doc = pymupdf.open(path)
    page = doc[page_no]
    H = page.rect.height
    W = page.rect.width
    d = Drawing(source="pdf")

    # --- text (whole lines, not single words) ---
    title_boxes = []
    for b in page.get_text("dict")["blocks"]:
        for line in b.get("lines", []):
            txt = "".join(sp["text"] for sp in line["spans"]).strip()
            if not txt:
                continue
            x0, y0, x1, y1 = line["bbox"]
            size = line["spans"][0]["size"]
            d.texts.append(Text(txt, (x0 + x1) / 2 * PT_TO_MM, (H - (y0 + y1) / 2) * PT_TO_MM, size * PT_TO_MM))
            if TITLE_KEYS.search(txt):
                title_boxes.append((x0, y0, x1, y1))

    # Title block region = union of keyword text boxes, grown a little.
    tb = None
    if title_boxes:
        tb = (min(b[0] for b in title_boxes) - 40, min(b[1] for b in title_boxes) - 15,
              max(b[2] for b in title_boxes) + 200, max(b[3] for b in title_boxes) + 15)

    paths = page.get_drawings()
    widths = [p.get("width") or 0 for p in paths if p.get("width")]
    wmax = max(widths) if widths else 1.0
    # Thick class = outline, by ISO 128 (0.5/0.7 mm vs 0.25/0.35 mm).
    thick_cut = 0.7 * wmax

    def inside_tb(x, y):
        return tb is not None and tb[0] <= x <= tb[2] and tb[1] <= y <= tb[3]

    for p in paths:
        r = p["rect"]
        if r.width * r.height > 0.5 * W * H:              # sheet frame
            continue
        kind = _dash_kind(p.get("dashes"))
        width = p.get("width") or 0
        if kind == "visible" and width < thick_cut:
            kind = "thin"                                  # dimension / leader / hatch
        items = p["items"]
        # closed 4-curve path with square bbox = circle; 3-curve = 3/4 thread arc
        if items and all(it[0] == "c" for it in items) and abs(r.width - r.height) < 0.02 * max(r.width, 1):
            n = len(items)
            if n >= 3:
                cx, cy = (r.x0 + r.x1) / 2, (r.y0 + r.y1) / 2
                if inside_tb(cx, cy):
                    continue
                d.circles.append(Circle(cx * PT_TO_MM, (H - cy) * PT_TO_MM, r.width / 2 * PT_TO_MM,
                                        kind if n >= 4 else "thin", partial=(n == 3)))
                continue
        pts_poly = []
        for it in items:
            if it[0] == "l":
                pts = [(it[1].x, it[1].y), (it[2].x, it[2].y)]
            elif it[0] == "c":
                pts = _bezier(*it[1:5])
            elif it[0] == "re":
                q = it[1]
                pts = [(q.x0, q.y0), (q.x1, q.y0), (q.x1, q.y1), (q.x0, q.y1), (q.x0, q.y0)]
            elif it[0] == "qu":
                q = it[1]
                pts = [(q.ul.x, q.ul.y), (q.ur.x, q.ur.y), (q.lr.x, q.lr.y), (q.ll.x, q.ll.y), (q.ul.x, q.ul.y)]
            else:
                continue
            for a, b in zip(pts, pts[1:]):
                if math.dist(a, b) < 1e-3:
                    continue
                if inside_tb(*a) and inside_tb(*b):
                    continue
                d.segs.append(Seg(a[0] * PT_TO_MM, (H - a[1]) * PT_TO_MM, b[0] * PT_TO_MM, (H - b[1]) * PT_TO_MM, kind))
            pts_poly += pts
        if p.get("closePath") and kind == "visible" and len(pts_poly) > 3:
            d.polys.append([[x * PT_TO_MM, (H - y) * PT_TO_MM] for x, y in pts_poly])

    d.notes.append("PDF geometry is in paper mm; scale applied from title block or calibrated from dimensions")
    return d
