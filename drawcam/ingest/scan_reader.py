"""Scanned / photographed drawings. This is the least reliable path and every
spec built from it is marked for review.

Pipeline: binarise -> OCR text with a pretrained LSTM (Tesseract) -> mask text ->
separate thick (visible outline) strokes from thin (dimension) strokes by stroke
width -> skeletonise -> Hough lines / circles -> merge -> paper millimetres.
"""
from __future__ import annotations

import math

import cv2
import numpy as np
import pytesseract

from .drawing import Drawing, Seg, Circle, Text


def _load(path_or_img, dpi):
    """Any raster input -> clean upright grey image at ~300 dpi (see image_prep)."""
    from .image_prep import load_image, prepare
    if isinstance(path_or_img, np.ndarray):
        img, file_dpi = path_or_img, dpi
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        try:
            img, file_dpi = load_image(path_or_img)
        except Exception as ex:
            raise ValueError(f"cannot open this image ({ex}). Save it as PNG or JPG and try again.") from None
    if min(img.shape) < 300:
        raise ValueError(f"image is only {img.shape[1]}x{img.shape[0]} pixels: far too small to read. "
                         "Scan at 300 dpi or photograph the whole sheet in good light.")
    return prepare(img, file_dpi)


def _stroke_widths(bw):
    dist = cv2.distanceTransform(bw, cv2.DIST_L2, 3)
    skel = cv2.ximgproc.thinning(bw)
    w = 2 * dist[skel > 0]
    return w[w > 0.5]


KEYWORDS = {"DP", "DEPTH", "DEEP", "THRU", "THROUGH", "POCKET", "DRILL", "TAP", "REAM", "GROOVE", "WIDE", "SLOT",
            "CBORE", "SPOTFACE", "THK", "THICK", "MATERIAL", "SCALE", "QTY", "PART", "NAME", "NO", "H7", "PCD", "TYP"}


def _ocr(gray):
    """Line-level OCR. Returns [(text, x0, y0, x1, y1, conf)] in pixels."""
    data = pytesseract.image_to_data(gray, config="--psm 11 --oem 1", output_type=pytesseract.Output.DICT)
    words = []
    for i, t in enumerate(data["text"]):
        t = (t or "").strip()
        conf = float(data["conf"][i])
        if not t or conf < 35:
            continue
        x, y, w, h = data["left"][i], data["top"][i], data["width"][i], data["height"][i]
        if h < 6 or (len(t) == 1 and not t.isalnum()):
            continue
        # small circles and arcs get "read" as O, o, 0, @, oe, so... - leave them as geometry
        if not any(ch.isdigit() for ch in t):
            if set(t) <= set("oO0Q@©°easo()") or (conf < 50 and t.upper().strip(".:") not in KEYWORDS):
                continue
        words.append([t, x, y, x + w, y + h, conf / 100])
    # Merge words that sit on the same baseline and are close together.
    words.sort(key=lambda w: (round(w[2] / 10), w[1]))
    lines = []
    for w in words:
        if lines:
            L = lines[-1]
            hgt = max(L[4] - L[2], 1)
            if abs(w[2] - L[2]) < 0.6 * hgt and 0 <= w[1] - L[3] < 1.2 * hgt:
                L[0] += " " + w[0]
                L[3], L[4] = max(L[3], w[3]), max(L[4], w[4])
                L[5] = min(L[5], w[5])
                continue
        lines.append(list(w))
    return lines


def _merge_segments(segs, ang_tol=2.0, dist_tol=3.0, gap=6.0):
    """Join collinear, overlapping Hough fragments (pixel units)."""
    out = []
    used = [False] * len(segs)
    for i, s in enumerate(segs):
        if used[i]:
            continue
        x1, y1, x2, y2 = s
        a = math.degrees(math.atan2(y2 - y1, x2 - x1)) % 180
        group = [s]
        used[i] = True
        for j in range(i + 1, len(segs)):
            if used[j]:
                continue
            u1, v1, u2, v2 = segs[j]
            b = math.degrees(math.atan2(v2 - v1, u2 - u1)) % 180
            if min(abs(a - b), 180 - abs(a - b)) > ang_tol:
                continue
            # perpendicular distance of segment j's midpoint from line i
            L = math.hypot(x2 - x1, y2 - y1) or 1
            mx, my = (u1 + u2) / 2, (v1 + v2) / 2
            dist = abs((x2 - x1) * (y1 - my) - (x1 - mx) * (y2 - y1)) / L
            if dist > dist_tol:
                continue
            # projected overlap / gap along the line
            dx, dy = (x2 - x1) / L, (y2 - y1) / L
            p = sorted([(x1 - x1) * dx + (y1 - y1) * dy, (x2 - x1) * dx + (y2 - y1) * dy])
            q = sorted([(u1 - x1) * dx + (v1 - y1) * dy, (u2 - x1) * dx + (v2 - y1) * dy])
            if q[0] > p[1] + gap or q[1] < p[0] - gap:
                continue
            group.append(segs[j])
            used[j] = True
        pts = np.array([[g[0], g[1]] for g in group] + [[g[2], g[3]] for g in group], float)
        d = np.array([math.cos(math.radians(a)), math.sin(math.radians(a))])
        proj = pts @ d
        c = pts.mean(axis=0)
        lo, hi = proj.min() - c @ d, proj.max() - c @ d
        p1, p2 = c + d * lo, c + d * hi
        # snap nearly axis-aligned lines exactly
        if a < 1.5 or a > 178.5:
            p1[1] = p2[1] = c[1]
        elif abs(a - 90) < 1.5:
            p1[0] = p2[0] = c[0]
        out.append((p1[0], p1[1], p2[0], p2[1]))
    return out


def read_scan(path_or_img, dpi: int = 300) -> Drawing:
    gray, dpi, prep_notes = _load(path_or_img, dpi)
    H, W = gray.shape
    px2mm = 25.4 / dpi
    d = Drawing(source="scan", scale_known=False)
    d.notes += prep_notes

    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if cv2.countNonZero(bw) > 0.35 * H * W:
        raise ValueError("the image is mostly dark: this does not look like a drawing on paper. "
                         "Photograph the sheet flat, in even light, filling most of the picture.")
    bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))   # speckle from JPEG/paper grain

    # 1) OCR text, then erase it so letters do not become "lines".
    #    Circles are found on the un-erased image (a hole next to its callout
    #    would otherwise lose its rim), excluding anything inside a text box.
    bw_full = bw.copy()
    text_boxes = []
    for t, x0, y0, x1, y1, conf in _ocr(gray):
        d.texts.append(Text(t, (x0 + x1) / 2 * px2mm, (H - (y0 + y1) / 2) * px2mm, (y1 - y0) * px2mm, conf))
        cv2.rectangle(bw, (x0 - 2, y0 - 2), (x1 + 2, y1 + 2), 0, -1)
        text_boxes.append((x0 - 3, y0 - 3, x1 + 3, y1 + 3))

    # 2) Remove sheet border: long lines hugging the image edge.
    margin = int(0.03 * min(H, W))
    frame = np.zeros_like(bw)
    frame[margin:H - margin, margin:W - margin] = 255
    frame_box = None
    contours, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    for c in contours:
        x, y, w, h = cv2.boundingRect(c)
        if w > 0.85 * W or h > 0.85 * H:
            frame_box = (x, y, x + w, y + h)
            cv2.drawContours(bw, [c], -1, 0, thickness=max(3, int(1.5 / px2mm)))

    # 3) Thick vs thin strokes. Opening with a kernel wider than a thin line
    #    keeps only outline (ISO 0.5-0.7 mm) strokes.
    #    The split width is checked against the strokes actually present: photos,
    #    hand drawings and some CAD plots use one line weight for everything.
    k = max(2, int(round(0.38 / px2mm)))
    sw = _stroke_widths(bw)
    uniform = False
    if len(sw) > 500:
        thin_w, thick_w = float(np.percentile(sw, 10)), float(np.percentile(sw, 90))
        if thick_w < 1.7 * thin_w:
            uniform = True
        elif not (thin_w < k < thick_w):
            k = max(2, int(round(math.sqrt(thin_w * thick_w))))
    if uniform:
        thick = bw.copy()
        thin = np.zeros_like(bw)
        d.notes.append("all lines have the same thickness: outline and dimension lines can't be told apart, check the profile carefully")
    else:
        thick = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k)))
        thick = cv2.dilate(thick, np.ones((3, 3), np.uint8))
        thin = cv2.subtract(bw, thick)

    # 4) Circles on the thick layer: every closed stroke gives an outer and an
    #    inner contour; fit a circle to each (least squares) and average the pair.
    found = []
    kk = 1 if uniform else k
    thick_full = cv2.morphologyEx(bw_full, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk)))
    thick_masked = cv2.morphologyEx(bw, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kk, kk)))
    passes = []
    # Circles are looked for on the outline layer AND on all strokes: blur in photos can
    # thin an outline circle below the split width. A fit must still be round and
    # closed, so dimension lines, arrows and partial thread arcs don't qualify.
    for img_, need_ in ((thick_full, 0.95), (thick_masked, 0.75), (bw_full, 0.97)):
        cs_, _ = cv2.findContours(img_, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
        passes += [(c, need_) for c in cs_]
    for c, need in passes:
        if len(c) < 20:
            continue
        P = c.reshape(-1, 2).astype(np.float64)
        A = np.c_[2 * P, np.ones(len(P))]
        bvec = (P ** 2).sum(1)
        sol, *_ = np.linalg.lstsq(A, bvec, rcond=None)
        cx, cy = sol[0], sol[1]
        r = math.sqrt(max(sol[2] + cx * cx + cy * cy, 0))
        if not (0.8 / px2mm < r < 90 / px2mm):
            continue
        res = np.abs(np.hypot(P[:, 0] - cx, P[:, 1] - cy) - r)
        ang = np.arctan2(P[:, 1] - cy, P[:, 0] - cx)
        coverage = len(np.unique(np.floor((ang + math.pi) / (2 * math.pi) * 36))) / 36
        in_text = any(bx0 <= cx - r and cx + r <= bx1 and by0 <= cy - r and cy + r <= by1 for bx0, by0, bx1, by1 in text_boxes)
        if res.mean() < max(1.5, 0.02 * r) and coverage > need and not in_text:
            found.append((cx, cy, r))
    merged = []
    stroke = k + 6
    for cx, cy, r in sorted(found, key=lambda t: t[2]):
        for m in merged:
            if math.hypot(m[0] - cx, m[1] - cy) < 3 and abs(m[2] - r) < stroke:
                m[3].append(r)
                break
        else:
            merged.append([cx, cy, r, [r]])
    for cx, cy, r, rs in merged:
        rr = (min(rs) + max(rs)) / 2
        d.circles.append(Circle(cx * px2mm, (H - cy) * px2mm, rr * px2mm))
        cv2.circle(thick, (int(round(cx)), int(round(cy))), int(round(rr)), 0, stroke)

    # 4b) Closed regions bounded by outline strokes (pockets, part outline). Their
    #     inner edge, grown by half a stroke, is the drawn centre line. Polygons
    #     survive broken corners far better than re-joining Hough fragments.
    try:
        from shapely.geometry import Polygon as _Poly, Point as _Pt
        dist = cv2.distanceTransform(thick_full, cv2.DIST_L2, 3)
        half = float(np.percentile(dist[dist > 0], 90)) if (dist > 0).any() else k / 2
        cs_, hier = cv2.findContours(thick_full, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)
        circ_mm = [(c.cx, c.cy, c.r) for c in d.circles]
        for i, c in enumerate(cs_):
            if hier is None or hier[0][i][3] == -1:          # only holes (enclosed faces)
                continue
            if cv2.contourArea(c) < (3 / px2mm) ** 2:
                continue
            ap = cv2.approxPolyDP(c, max(1.0, 0.15 / px2mm), True).reshape(-1, 2)
            if len(ap) < 3:
                continue
            poly = _Poly([(x * px2mm, (H - y) * px2mm) for x, y in ap]).buffer(half * px2mm, join_style=2)
            if poly.is_empty or not poly.is_valid:
                continue
            # skip faces that are just the inside of a detected circle
            if any(abs(poly.area - math.pi * r * r) < 0.08 * math.pi * r * r and poly.centroid.distance(_Pt(cx, cy)) < 0.3 * r
                   for cx, cy, r in circ_mm):
                continue
            d.polys.append([list(p_) for p_ in poly.exterior.coords])
    except Exception as ex:                                       # polygons are a bonus, never fatal
        d.notes.append(f"region extraction skipped: {ex}")

    # 5) Lines: skeleton + probabilistic Hough + merge.
    skel = cv2.ximgproc.thinning(thick)
    minlen = int(1.5 / px2mm)
    raw = cv2.HoughLinesP(skel, 1, np.pi / 360, threshold=max(10, minlen // 2), minLineLength=minlen, maxLineGap=int(0.8 / px2mm))
    if raw is not None:
        for x1, y1, x2, y2 in _merge_segments([tuple(map(float, l)) for l in raw.reshape(-1, 4)], gap=0.8 / px2mm):
            d.segs.append(Seg(x1 * px2mm, (H - y1) * px2mm, x2 * px2mm, (H - y2) * px2mm, "visible"))

    # 6) Centre lines: long, dashed, thin, roughly horizontal or vertical.
    raw = cv2.HoughLinesP(thin, 1, np.pi / 180, threshold=60, minLineLength=int(0.25 * W), maxLineGap=int(4 / px2mm))
    if raw is not None:
        for x1, y1, x2, y2 in _merge_segments([tuple(map(float, l)) for l in raw.reshape(-1, 4)], gap=4 / px2mm):
            if abs(y2 - y1) < 3 or abs(x2 - x1) < 3:
                d.segs.append(Seg(x1 * px2mm, (H - y1) * px2mm, x2 * px2mm, (H - y2) * px2mm, "center"))

    # 7) Drop sheet frame and title block geometry.
    import re as _re
    keys = _re.compile(r"MATERIAL|SCALE|DRG|DRAWN|QTY|TITLE|CHECKED|PART\s*N|DATE|THK", _re.I)
    tb = [t for t in d.texts if keys.search(t.text)]
    tbox = None
    if tb:
        x0 = min(t.x for t in tb) - 15
        y1 = max(t.y for t in tb) + 6
        tbox = (x0, -1e9, 1e9, y1)
    fb = None
    if frame_box:
        fx0, fy0, fx1, fy1 = frame_box
        fb = (fx0 * px2mm, (H - fy1) * px2mm, fx1 * px2mm, (H - fy0) * px2mm)

    def junk(xa, ya, xb, yb):
        if tbox and all(tbox[0] <= x <= tbox[2] and tbox[1] <= y <= tbox[3] for x, y in ((xa, ya), (xb, yb))):
            return True
        if fb:
            m = 2.5
            near = lambda x, y: (abs(x - fb[0]) < m or abs(x - fb[2]) < m or abs(y - fb[1]) < m or abs(y - fb[3]) < m)
            if near(xa, ya) and near(xb, yb) and (abs(xa - xb) < m or abs(ya - yb) < m):
                return True
        return False
    d.segs = [s_ for s_ in d.segs if not junk(s_.x1, s_.y1, s_.x2, s_.y2)]
    d.circles = [c for c in d.circles if not junk(c.cx, c.cy, c.cx, c.cy)]
    if fb:
        fw, fh = fb[2] - fb[0], fb[3] - fb[1]
        d.polys = [p_ for p_ in d.polys if not (max(x for x, _ in p_) - min(x for x, _ in p_) > 0.8 * fw)]
    if tbox:
        d.polys = [p_ for p_ in d.polys if not all(tbox[0] <= x <= tbox[2] and tbox[1] <= y <= tbox[3] for x, y in p_)]

    d.notes.append(f"image read at ~{dpi:.0f} dpi: geometry is approximate, snapped to OCR dimensions where possible")
    return d


def pdf_page_to_image(path: str, dpi: int = 300, page_no: int = 0):
    import pymupdf
    page = pymupdf.open(path)[page_no]
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    return np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width)
