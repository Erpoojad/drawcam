"""Turns any raster drawing (flatbed scan, phone photo, screenshot, scanned PDF
page) into a clean, upright, evenly lit greyscale image at a known resolution,
so the scan reader sees roughly the same thing every time.

Steps
  1. load: EXIF rotation, transparency -> white, CMYK/16-bit/palette -> 8-bit grey
  2. find the sheet in a photo and remove perspective (4-corner warp)
  3. flatten lighting (divide by a blurred background estimate: kills shadows)
  4. deskew small rotations from the dominant line directions
  5. resample to an effective 300 dpi, assuming the sheet is an A-size sheet
     (only used for size thresholds: real model scale comes from the dimensions)
"""
from __future__ import annotations

import math

import cv2
import numpy as np

TARGET_DPI = 300
A_SIZES = {"A4": (297, 210), "A3": (420, 297), "A2": (594, 420), "A1": (841, 594), "A0": (1189, 841)}
MAX_SIDE = 5000           # pixels after resampling (memory on small servers)


def load_image(path: str):
    """Returns (grey uint8, dpi or None). Handles what phones and scanners produce."""
    from PIL import Image, ImageOps
    Image.MAX_IMAGE_PIXELS = 200_000_000
    im = Image.open(path)
    try:
        im.seek(0)                                    # first page of multi-page TIFF
    except EOFError:
        pass
    dpi = im.info.get("dpi")
    dpi = float(dpi[0]) if dpi and dpi[0] and 50 <= float(dpi[0]) <= 2400 else None
    im = ImageOps.exif_transpose(im)                  # phone photos are often stored sideways
    if im.mode in ("RGBA", "LA", "PA") or (im.mode == "P" and "transparency" in im.info):
        im = im.convert("RGBA")
        bg = Image.new("RGBA", im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(bg, im)
    if im.mode in ("I;16", "I;16B", "I;16L", "I"):
        a = np.asarray(im, dtype=np.float64)
        a = (255 * (a - a.min()) / max(np.ptp(a), 1)).astype(np.uint8)
        return a, dpi
    if im.mode == "1":
        im = im.convert("L")
    return np.asarray(im.convert("L")), dpi


def _order(pts):
    pts = np.asarray(pts, np.float32)
    s, d = pts.sum(1), np.diff(pts, axis=1).ravel()
    return np.array([pts[s.argmin()], pts[d.argmin()], pts[s.argmax()], pts[d.argmax()]], np.float32)


def find_sheet(gray):
    """Bright 4-sided region = the paper in a photo. Returns 4 corners or None."""
    h, w = gray.shape
    scale = 1000 / max(h, w)
    small = cv2.resize(gray, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA) if scale < 1 else gray.copy()
    scale = min(scale, 1.0)
    blur = cv2.GaussianBlur(small, (7, 7), 0)
    _, mask = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))   # fill the drawing's lines
    cs, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return None
    c = max(cs, key=cv2.contourArea)
    area = cv2.contourArea(c) / (small.shape[0] * small.shape[1])
    if not 0.15 < area < 0.97:                         # whole image is paper (scan) or no paper found
        return None
    peri = cv2.arcLength(c, True)
    for eps in (0.01, 0.02, 0.03, 0.05):
        ap = cv2.approxPolyDP(c, eps * peri, True)
        if len(ap) == 4 and cv2.isContourConvex(ap):
            return _order(ap.reshape(4, 2) / scale)
    return None


def warp_sheet(gray, quad):
    tl, tr, br, bl = quad
    wid = max(np.linalg.norm(tr - tl), np.linalg.norm(br - bl))
    hgt = max(np.linalg.norm(bl - tl), np.linalg.norm(br - tr))
    # paper sheets are 1:sqrt(2); snap if close (perspective distorts the ratio)
    r = max(wid, hgt) / max(min(wid, hgt), 1)
    if abs(r - math.sqrt(2)) < 0.25:
        if wid >= hgt:
            hgt = wid / math.sqrt(2)
        else:
            wid = hgt / math.sqrt(2)
    W, H = int(round(wid)), int(round(hgt))
    M = cv2.getPerspectiveTransform(quad, np.float32([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]]))
    return cv2.warpPerspective(gray, M, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)


def rectify_by_frame(gray):
    """A drawing sheet's border frame is a precise rectangle. If its corners are
    found, warp so it is exactly rectangular: removes the residual perspective
    the paper-edge warp leaves (~0.5 mm), and any skew. Returns (img, done)."""
    h, w = gray.shape
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cs, _ = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cs:
        return gray, False
    c = max(cs, key=lambda c: cv2.boundingRect(c)[2] * cv2.boundingRect(c)[3])
    x, y, bw_, bh = cv2.boundingRect(c)
    if bw_ * bh < 0.55 * w * h:
        return gray, False
    hull = cv2.convexHull(c)
    peri = cv2.arcLength(hull, True)
    quad = None
    for eps in (0.005, 0.01, 0.02):
        ap = cv2.approxPolyDP(hull, eps * peri, True)
        if len(ap) == 4:
            quad = _order(ap.reshape(4, 2).astype(np.float32))
            break
    if quad is None:
        return gray, False
    # refine each corner to sub-pixel accuracy by fitting lines to the frame edges
    edges = []
    for i in range(4):
        a, b = quad[i], quad[(i + 1) % 4]
        pts = c.reshape(-1, 2).astype(np.float32)
        ab = b - a
        L = np.linalg.norm(ab)
        t = ((pts - a) @ ab) / (L * L)
        rel = pts - a
        dist = np.abs(ab[0] * rel[:, 1] - ab[1] * rel[:, 0]) / L
        sel = pts[(t > 0.1) & (t < 0.9) & (dist < 6)]
        if len(sel) < 20:
            return gray, False
        vx, vy, x0, y0 = cv2.fitLine(sel, cv2.DIST_HUBER, 0, 0.01, 0.01).ravel()
        edges.append((np.array([x0, y0]), np.array([vx, vy])))
    corners = []
    for i in range(4):
        (p1, d1), (p2, d2) = edges[i - 1], edges[i]
        A = np.array([d1, -d2]).T
        if abs(np.linalg.det(A)) < 1e-6:
            return gray, False
        tt = np.linalg.solve(A, p2 - p1)
        corners.append(p1 + d1 * tt[0])
    src = np.float32(corners)
    tl, tr, br, bl = src
    W = (np.linalg.norm(tr - tl) + np.linalg.norm(br - bl)) / 2
    H = (np.linalg.norm(bl - tl) + np.linalg.norm(br - tr)) / 2
    dst = np.float32([tl, tl + [W, 0], tl + [W, H], tl + [0, H]])
    move = np.abs(src - dst).max()
    if move < 0.7:
        return gray, False                            # already square to well under a pixel
    if move > 0.03 * max(w, h):
        return gray, False                            # a big correction means the "frame" is something else
    for i in range(4):                                # all four corners must be close to 90 degrees
        a, b, c2 = src[i - 1], src[i], src[(i + 1) % 4]
        u, v = a - b, c2 - b
        cosang = abs(float(u @ v) / (np.linalg.norm(u) * np.linalg.norm(v)))
        if cosang > 0.05:
            return gray, False
    M = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=255), True


def flatten_light(gray):
    """Divide out uneven illumination; result has a white background."""
    h, w = gray.shape
    k = max(31, (min(h, w) // 25) | 1)
    bg = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15)))
    bg = cv2.medianBlur(cv2.resize(bg, (w // 4 or 1, h // 4 or 1), interpolation=cv2.INTER_AREA), (k // 4) | 1)
    bg = cv2.resize(bg, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32)
    out = gray.astype(np.float32) / np.maximum(bg, 1) * 255
    out = np.clip(out, 0, 255)
    # stretch so ink is dark and paper is white
    lo, hi = np.percentile(out, 1), np.percentile(out, 60)
    out = np.clip((out - lo) / max(hi - lo, 1) * 255, 0, 255)
    return out.astype(np.uint8)


def deskew(gray):
    """Rotate by the median deviation of long lines from horizontal/vertical (<= 8 deg)."""
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    h, w = bw.shape
    lines = cv2.HoughLinesP(bw, 1, np.pi / 720, threshold=150, minLineLength=min(h, w) // 6, maxLineGap=5)
    if lines is None:
        return gray, 0.0
    devs = []
    for x1, y1, x2, y2 in lines.reshape(-1, 4):
        a = math.degrees(math.atan2(y2 - y1, x2 - x1))
        dev = ((a + 45) % 90) - 45
        if abs(dev) < 8:
            devs.append(dev)
    if len(devs) < 5:
        return gray, 0.0
    ang = float(np.median(devs))
    if abs(ang) < 0.15:
        return gray, 0.0
    M = cv2.getRotationMatrix2D((w / 2, h / 2), ang, 1.0)
    return cv2.warpAffine(gray, M, (w, h), flags=cv2.INTER_CUBIC, borderValue=255), ang


def guess_sheet_mm(w_px, h_px, gray):
    """Which A-size sheet is this? Uses the line weight: ISO outlines are ~0.5-0.7 mm,
    so the sheet size that makes the thick lines closest to 0.6 mm wins."""
    long_px = max(w_px, h_px)
    _, bw = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dist = cv2.distanceTransform(bw, cv2.DIST_L2, 3)
    skel = cv2.ximgproc.thinning(bw)
    widths = 2 * dist[skel > 0]
    widths = widths[widths > 0.5]
    if len(widths) < 200:
        return "A4", A_SIZES["A4"]
    thick_px = float(np.percentile(widths, 90))
    best = min(A_SIZES.items(), key=lambda kv: abs(math.log((thick_px * kv[1][0] / long_px) / 0.6)))
    return best


def prepare(gray, dpi=None, is_photo=None):
    """Returns (clean grey image at ~TARGET_DPI, effective dpi, notes)."""
    notes = []
    h, w = gray.shape
    quad = find_sheet(gray) if is_photo is not False else None
    if quad is not None:
        gray = warp_sheet(gray, quad)
        notes.append("photo: sheet edges found and perspective corrected")
        dpi = None                                    # pixel size is meaningless after a warp
    gray = flatten_light(gray)
    gray, framed = rectify_by_frame(gray)
    if framed:
        notes.append("squared up using the sheet's border frame")
        ang = 0.0
    else:
        gray, ang = deskew(gray)
    if ang:
        notes.append(f"deskewed by {ang:.1f} deg")
    h, w = gray.shape
    if dpi is None or quad is not None:
        name, (Lmm, _) = guess_sheet_mm(w, h, gray)
        dpi = max(w, h) / (Lmm / 25.4)
        notes.append(f"no reliable dpi: assumed an {name} sheet ({dpi:.0f} dpi)")
    f = TARGET_DPI / dpi
    if max(h, w) * f > MAX_SIDE:
        f = MAX_SIDE / max(h, w)
    if abs(f - 1) > 0.05:
        gray = cv2.resize(gray, (int(w * f), int(h * f)), interpolation=cv2.INTER_CUBIC if f > 1 else cv2.INTER_AREA)
        dpi = dpi * f
    if dpi < 120:
        notes.append(f"low resolution (~{dpi:.0f} dpi effective): small text and holes may be missed")
    return gray, dpi, notes
