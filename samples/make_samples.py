"""Builds test drawings for three parts in three formats each:
DXF (CAD), vector PDF (CAD plot), and PNG (simulated scan with noise).

  shaft  - stepped shaft, M20x1.5 thread + relief groove, needs two setups (EN8)
  bush   - flanged bush with Ø20 H7 through bore (MS)
  plate  - 120x80x20 plate: holes, M8 taps, H7 ream, rectangular + round pocket (AL)
"""
from __future__ import annotations

import math
import os
import sys

import ezdxf
import numpy as np
import pymupdf
from reportlab.lib.pagesizes import A4, landscape
from reportlab.pdfgen import canvas

HERE = os.path.dirname(os.path.abspath(__file__))


class Part:
    def __init__(self, name):
        self.name = name
        self.segs = []      # (x1,y1,x2,y2,kind)  kind: visible|thin|center|hidden
        self.circles = []   # (cx,cy,r,kind,partial)
        self.texts = []     # (text,x,y,h)
        self.title = {}

    def L(self, x1, y1, x2, y2, kind="visible"):
        self.segs.append((x1, y1, x2, y2, kind))

    def C(self, cx, cy, r, kind="visible", partial=False):
        self.circles.append((cx, cy, r, kind, partial))

    def T(self, t, x, y, h=3.5):
        self.texts.append((t, x, y, h))


def revolve_side(p: Part, x0, axis, profile, chamfer=1.0):
    """profile from the RIGHT end: [(length, dia), ...] drawn leftwards from x0."""
    x = x0
    pts = []
    for length, dia in profile:
        pts.append((x, dia))
        x -= length
        pts.append((x, dia))
    # outline above/below axis + vertical faces
    prev = None
    for i in range(0, len(pts), 2):
        (xa, d), (xb, _) = pts[i], pts[i + 1]
        r = d / 2
        for sgn in (1, -1):
            p.L(xa, axis + sgn * r, xb, axis + sgn * r)
        if prev is not None and prev != d:
            p.L(xa, axis - max(prev, d) / 2, xa, axis + max(prev, d) / 2)
        prev = d
    p.L(pts[0][0], axis - pts[0][1] / 2, pts[0][0], axis + pts[0][1] / 2)
    p.L(pts[-1][0], axis - pts[-1][1] / 2, pts[-1][0], axis + pts[-1][1] / 2)
    p.L(pts[-1][0] - 5, axis, pts[0][0] + 5, axis, "center")


def shaft():
    p = Part("shaft")
    # right->left: thread M20 x 22, relief 3 to Ø16.9, Ø25 x 32, Ø40 x 15, Ø30 x 28  (total 100)
    prof = [(22, 20.0), (3, 16.9), (32, 25.0), (15, 40.0), (28, 30.0)]
    revolve_side(p, 100, 0, prof)
    p.L(78, 8.16, 100, 8.16, "thin")       # thread minor line
    p.L(78, -8.16, 100, -8.16, "thin")
    # end view (right of side view)
    for r in (15, 20, 12.5, 10):
        p.C(135, 0, r)
    p.C(135, 0, 8.16, "thin", True)
    p.L(110, 0, 160, 0, "center")
    p.L(135, -25, 135, 25, "center")
    p.T("M20x1.5", 89, 16)
    p.T("Ø25 ±0.01", 62, 18)
    p.T("Ø40", 46, 26)
    p.T("Ø30", 14, 21)
    p.T("GROOVE 3 WIDE Ø16.9", 70, -24)
    p.T("100", 50, -34)
    p.T("22", 89, -18)
    p.T("C1", 102, 12)
    p.title = {"PART NAME": "SHAFT-01", "MATERIAL": "EN8", "QTY": "50"}
    return p


def bush():
    p = Part("bush")
    prof = [(30, 35.0), (10, 60.0)]          # Ø35 x 30 then flange Ø60 x 10
    revolve_side(p, 40, 0, prof)
    # bore Ø20 through as hidden lines
    p.L(0, 10, 40, 10, "hidden")
    p.L(0, -10, 40, -10, "hidden")
    for r in (30, 17.5):
        p.C(85, 0, r)
    p.C(85, 0, 10)
    p.L(50, 0, 120, 0, "center")
    p.L(85, -35, 85, 35, "center")
    p.T("Ø35 +0.0/-0.03", 22, 23)
    p.T("Ø60", 5, 35)
    p.T("Ø20 H7 THRU", 20, -26)
    p.T("40", 20, -40)
    p.T("C0.5", 42, 19)
    p.title = {"PART NAME": "BUSH-07", "MATERIAL": "MS", "QTY": "200"}
    return p


def plate():
    p = Part("plate")
    W, H, T = 120.0, 80.0, 20.0
    # top view, origin at bottom-left
    p.L(0, 0, W, 0); p.L(W, 0, W, H); p.L(W, H, 0, H); p.L(0, H, 0, 0)
    for x, y in ((10, 10), (110, 10), (10, 70), (110, 70)):
        p.C(x, y, 4.25)
    for x, y in ((30, 70), (90, 70)):
        p.C(x, y, 3.4)
        p.C(x, y, 4.0, "thin", True)
    p.C(100, 40, 5.0)
    p.C(20, 40, 15.0)                       # Ø30 round pocket
    # rect pocket 40x30 R5, centred (70,40)
    cx, cy, w, h, r = 62, 40, 40, 30, 5
    x0, x1, y0, y1 = cx - w / 2, cx + w / 2, cy - h / 2, cy + h / 2
    p.L(x0 + r, y0, x1 - r, y0); p.L(x1, y0 + r, x1, y1 - r); p.L(x1 - r, y1, x0 + r, y1); p.L(x0, y1 - r, x0, y0 + r)
    for (ccx, ccy, a0) in ((x1 - r, y0 + r, -90), (x1 - r, y1 - r, 0), (x0 + r, y1 - r, 90), (x0 + r, y0 + r, 180)):
        for k in range(8):
            a = math.radians(a0 + 90 * k / 8)
            b = math.radians(a0 + 90 * (k + 1) / 8)
            p.L(ccx + r * math.cos(a), ccy + r * math.sin(a), ccx + r * math.cos(b), ccy + r * math.sin(b))
    # side view below
    p.L(0, -40, W, -40); p.L(W, -40, W, -40 + T); p.L(W, -40 + T, 0, -40 + T); p.L(0, -40 + T, 0, -40)
    p.L(x0, -20, x0, -28, "hidden"); p.L(x1, -20, x1, -28, "hidden"); p.L(x0, -28, x1, -28, "hidden")
    p.T("4X Ø8.5 THRU", 18, 2)
    p.T("2X M8 DP 15", 60, 75)
    p.T("Ø10 H7 THRU", 104, 50)
    p.T("Ø30 DP 5", 20, 60)
    p.T("POCKET DP 8", 62, 40)
    p.T("120", 60, 88)
    p.T("80", -10, 40)
    p.T("20", 128, -30)
    p.title = {"PART NAME": "PLATE-12", "MATERIAL": "AL 6061", "QTY": "20", "THK": "20"}
    return p


# ------------------------------------------------------------------ writers
def to_dxf(p: Part, path):
    doc = ezdxf.new("R2010", setup=True)
    doc.header["$INSUNITS"] = 4
    doc.layers.add("OUTLINE", lineweight=50)
    doc.layers.add("DIM", lineweight=25)
    doc.layers.add("CENTER", linetype="CENTER", lineweight=25)
    doc.layers.add("HIDDEN", linetype="HIDDEN", lineweight=25)
    msp = doc.modelspace()
    layer = {"visible": "OUTLINE", "thin": "DIM", "center": "CENTER", "hidden": "HIDDEN"}
    for x1, y1, x2, y2, k in p.segs:
        msp.add_line((x1, y1), (x2, y2), dxfattribs={"layer": layer[k]})
    for cx, cy, r, k, partial in p.circles:
        if partial:
            msp.add_arc((cx, cy), r, 0, 270, dxfattribs={"layer": layer[k]})
        else:
            msp.add_circle((cx, cy), r, dxfattribs={"layer": layer[k]})
    for t, x, y, h in p.texts:
        msp.add_text(t, height=h, dxfattribs={"layer": "DIM"}).set_placement((x, y))
    y = -60
    for k, v in p.title.items():
        msp.add_text(f"{k}: {v}", height=3.5, dxfattribs={"layer": "TITLE"}).set_placement((150, y))
        y -= 6
    doc.saveas(path)


def to_pdf(p: Part, path, scale=1.0):
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    font = "Helvetica"
    for cand in ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/usr/share/fonts/dejavu/DejaVuSans.ttf"):
        if os.path.exists(cand):
            pdfmetrics.registerFont(TTFont("DejaVu", cand))
            font = "DejaVu"
            break
    W, H = landscape(A4)
    mm = 72 / 25.4
    c = canvas.Canvas(path, pagesize=(W, H))
    xs = [v for s in p.segs for v in (s[0], s[2])] + [ci[0] - ci[2] for ci in p.circles] + [ci[0] + ci[2] for ci in p.circles]
    ys = [v for s in p.segs for v in (s[1], s[3])] + [ci[1] - ci[2] for ci in p.circles] + [ci[1] + ci[2] for ci in p.circles]
    ox = 25 - min(xs) * scale
    oy = 60 + (150 - (max(ys) - min(ys)) * scale) / 2 - min(ys) * scale
    P = lambda x, y: ((ox + x * scale) * mm, (oy + y * scale) * mm)
    style = {"visible": (0.6, None), "thin": (0.25, None), "center": (0.25, [12, 2, 2, 2]), "hidden": (0.3, [4, 2])}
    c.setLineWidth(0.7 * mm); c.rect(8 * mm, 8 * mm, W - 16 * mm, H - 16 * mm)     # frame
    for x1, y1, x2, y2, k in p.segs:
        lw, dash = style[k]
        c.setLineWidth(lw * mm); c.setDash(dash) if dash else c.setDash()
        c.line(*P(x1, y1), *P(x2, y2))
    for cx, cy, r, k, partial in p.circles:
        lw, dash = style[k]
        c.setLineWidth(lw * mm); c.setDash(dash) if dash else c.setDash()
        X, Y = P(cx, cy)
        R = r * scale * mm
        if partial:
            c.arc(X - R, Y - R, X + R, Y + R, 0, 270)
        else:
            c.circle(X, Y, R)
    c.setDash()
    for t, x, y, h in p.texts:
        c.setFont(font, 3.5 * mm)
        X, Y = P(x, y)
        c.drawCentredString(X, Y, t)
    # title block
    c.setLineWidth(0.35 * mm)
    tx, ty = W - 120 * mm, 8 * mm
    c.rect(tx, ty, 112 * mm, 34 * mm)
    rows = [f"{k}: {v}" for k, v in p.title.items() if k != "THK"] + [f"SCALE {scale:g}:1" if scale >= 1 else f"SCALE 1:{1/scale:g}"]
    if "THK" in p.title:
        rows.append(f"THK {p.title['THK']}")
    c.setFont(font, 3.5 * mm)
    for i, row in enumerate(rows):
        c.drawString(tx + 3 * mm, ty + (28 - 6 * i) * mm, row)
    c.save()


def to_scan(pdf_path, png_path, dpi=300, seed=0):
    page = pymupdf.open(pdf_path)[0]
    pix = page.get_pixmap(dpi=dpi, colorspace=pymupdf.csGRAY)
    img = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width).astype(np.float32)
    rng = np.random.default_rng(seed)
    import cv2
    img = cv2.GaussianBlur(img, (3, 3), 0.8)
    img = img * rng.uniform(0.88, 0.96) + rng.normal(0, 9, img.shape) + 12
    img = np.clip(img, 0, 255).astype(np.uint8)
    cv2.imwrite(png_path, img)


def main(outdir=HERE):
    os.makedirs(outdir, exist_ok=True)
    for part, scale in ((shaft(), 1.5), (bush(), 2.0), (plate(), 1.0)):
        base = os.path.join(outdir, part.name)
        to_dxf(part, base + ".dxf")
        to_pdf(part, base + ".pdf", scale)
        to_scan(base + ".pdf", base + "_scan.png")
        print("wrote", base + ".{dxf,pdf,_scan.png}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else HERE)


# ------------------------------------------------- realistic raster inputs
def to_photo(pdf_path, out_path, seed=0, tilt=0.08, sideways=False):
    """Phone photo of the printed sheet: on a table, perspective, shadow, warm light, JPEG."""
    import cv2
    from PIL import Image
    rng = np.random.default_rng(seed)
    page = pymupdf.open(pdf_path)[0]
    pix = page.get_pixmap(dpi=220, colorspace=pymupdf.csRGB)
    sheet = np.frombuffer(pix.samples, np.uint8).reshape(pix.height, pix.width, 3).astype(np.float32)
    h, w = sheet.shape[:2]
    H, W = int(h * 1.35), int(w * 1.3)
    table = np.full((H, W, 3), (70, 85, 105), np.float32) + rng.normal(0, 6, (H, W, 3))
    j = lambda s: rng.uniform(-s, s)
    ox, oy = (W - w) / 2, (H - h) / 2
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[ox + j(tilt * w), oy + j(tilt * h)], [ox + w + j(tilt * w), oy + j(tilt * h)],
                      [ox + w + j(tilt * w), oy + h + j(tilt * h)], [ox + j(tilt * w), oy + h + j(tilt * h)]])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(sheet, M, (W, H), flags=cv2.INTER_LINEAR)
    mask = cv2.warpPerspective(np.ones((h, w), np.float32), M, (W, H))[..., None]
    img = table * (1 - mask) + warped * mask
    yy, xx = np.mgrid[0:H, 0:W]
    shade = 1 - 0.38 * np.clip((xx / W) * 0.8 + (yy / H) * 0.6 - 0.35, 0, 1)        # shadow across one corner
    img = img * shade[..., None] * np.array([1.0, 0.96, 0.88])                       # warm indoor light
    img = cv2.GaussianBlur(img, (3, 3), 0.9) + rng.normal(0, 4, img.shape)
    img = np.clip(img, 0, 255).astype(np.uint8)
    im = Image.fromarray(img)
    if sideways:                                    # stored rotated, EXIF says "rotate 90"
        im = im.transpose(Image.Transpose.ROTATE_90)
        ex = im.getexif()
        ex[0x0112] = 6
        im.save(out_path, "JPEG", quality=72, exif=ex)
    else:
        im.save(out_path, "JPEG", quality=72)


def to_screenshot(pdf_path, out_path):
    """Low-resolution screen grab of the drawing (no dpi metadata)."""
    page = pymupdf.open(pdf_path)[0]
    page.get_pixmap(dpi=130).save(out_path)


def to_scanned_pdf(png_path, out_path):
    """Scanner 'scan to PDF' with an invisible OCR text layer on top of the picture."""
    doc = pymupdf.open()
    page = doc.new_page(width=842, height=595)
    page.insert_image(page.rect, filename=png_path)
    page.insert_text((60, 60), "MATERIAL SCALE DRAWN CHECKED", fontsize=9, render_mode=3)
    doc.save(out_path)


def make_raster_set(outdir):
    out = os.path.join(outdir, "raster")
    os.makedirs(out, exist_ok=True)
    for i, name in enumerate(("shaft", "bush", "plate")):
        pdf = os.path.join(outdir, name + ".pdf")
        to_photo(pdf, os.path.join(out, f"{name}_photo.jpg"), seed=i)
        to_photo(pdf, os.path.join(out, f"{name}_photo_sideways.jpeg"), seed=10 + i, sideways=True)
        to_screenshot(pdf, os.path.join(out, f"{name}_screenshot.png"))
        to_scanned_pdf(os.path.join(outdir, name + "_scan.png"), os.path.join(out, f"{name}_scanned.pdf"))
    return out
