"""Drawing -> MillSpec. Reads the top view of a prismatic part: outline,
holes (drilled / tapped / reamed / interpolated), pockets; thickness from text
or from the side view."""
from __future__ import annotations

import math

from shapely.geometry import LineString, Polygon, Point
from shapely.ops import polygonize, unary_union

from ..ingest.drawing import Drawing
from ..ingest.dim_parser import COARSE_PITCH
from ..spec import MillSpec, Hole, Pocket, ReviewItem
from .common import split_views, top_view, callouts, globals_, nearest

MAX_DRILL = 20.0          # above this, holes are helically interpolated with an end mill


def _faces(segs, snap_tol):
    q = lambda v: round(v / snap_tol) * snap_tol
    lines = [LineString([(q(s.x1), q(s.y1)), (q(s.x2), q(s.y2))]) for s in segs if s.kind == "visible"]
    lines = [l for l in lines if l.length > 0]
    if not lines:
        return []
    u = unary_union(lines)
    polys = [Polygon(p.exterior) for p in polygonize(list(getattr(u, "geoms", [u])))]
    uniq = []
    for p in sorted(polys, key=lambda p: -p.area):
        if p.area > 1.0 and not any(abs(p.area - u.area) < 1e-3 * u.area and p.equals_exact(u, snap_tol) for u in uniq):
            uniq.append(p)
    return uniq


def _rect_info(poly: Polygon):
    """If poly is an axis-aligned rectangle (optionally with round corners) return (cx,cy,w,h,r)."""
    minx, miny, maxx, maxy = poly.bounds
    w, h = maxx - minx, maxy - miny
    deficit = w * h - poly.area
    if deficit < -1e-6 or deficit > 0.25 * w * h:
        return None
    r = math.sqrt(max(deficit, 0) / (4 - math.pi))
    if abs(r * 2 - round(r * 2)) < 0.2:          # drawings use round radii; chorded arcs read a bit large
        r = round(r * 2) / 2
    test = Polygon([(minx, miny), (maxx, miny), (maxx, maxy), (minx, maxy)])
    if test.buffer(-r - 0.05).within(poly) if r > 0 else abs(poly.area - test.area) < 1e-3 * test.area:
        return ((minx + maxx) / 2, (miny + maxy) / 2, w, h, round(r, 2))
    return None


def recognize_milling(d: Drawing) -> MillSpec:
    review = []
    views = split_views(d)
    top = top_view(views)
    others = [v for v in views if v is not top]
    cs = callouts(d)
    g = globals_(d)
    is_scan = d.source == "scan"
    snap_tol = 0.5 if is_scan else 0.01

    faces = _faces(top.segs, snap_tol)
    if is_scan:
        tb_ = top.bbox()
        regions = [Polygon(p_) for p_ in d.polys]
        regions = [r for r in regions if r.is_valid and r.within(Polygon([(tb_[0] - 2, tb_[1] - 2), (tb_[2] + 2, tb_[1] - 2), (tb_[2] + 2, tb_[3] + 2), (tb_[0] - 2, tb_[3] + 2)]))]
        if regions:
            faces = sorted(regions, key=lambda r: -r.area)
    tb = top.bbox()
    def snap_v(v, tol=0.2):
        if is_scan and abs(v * 2 - round(v * 2)) < 2 * tol:
            return round(v * 2) / 2
        return v

    if is_scan and faces:
        # an enclosed region that is round is a hole the circle finder missed
        from ..ingest.drawing import Circle
        keep = []
        for fpoly in faces:
            mnx, mny, mxx, mxy = fpoly.bounds
            w_, h_ = mxx - mnx, mxy - mny
            r_eq = math.sqrt(fpoly.area / math.pi)
            if abs(w_ - h_) < 0.08 * max(w_, h_) and abs(w_ / 2 - r_eq) < 0.06 * r_eq and r_eq < 0.45 * max(top.width, top.height):
                c0 = fpoly.centroid
                if not any(math.hypot(c.cx - c0.x, c.cy - c0.y) < 0.5 and abs(c.r - r_eq) < 0.5 for c in top.circles):
                    top.circles.append(Circle(c0.x, c0.y, r_eq))
                continue
            keep.append(fpoly)
        faces = keep
        # snap a near-rectangular outline to an exact rectangle
        if faces:
            o = faces[0]
            mnx, mny, mxx, mxy = o.bounds
            if o.area > 0.97 * (mxx - mnx) * (mxy - mny):
                faces[0] = Polygon([(snap_v(mnx), snap_v(mny)), (snap_v(mxx), snap_v(mny)), (snap_v(mxx), snap_v(mxy)), (snap_v(mnx), snap_v(mxy))])
    if faces:
        outline_poly = faces[0]
    else:
        outline_poly = Polygon([(tb[0], tb[1]), (tb[2], tb[1]), (tb[2], tb[3]), (tb[0], tb[3])])
        review.append(ReviewItem("outline", "no closed outline found, using the bounding rectangle"))
    ox, oy = outline_poly.bounds[0], outline_poly.bounds[1]
    W = outline_poly.bounds[2] - ox
    Hh = outline_poly.bounds[3] - oy

    # --- thickness ---
    thick = g["thickness"]
    if thick is None:
        side = [v for v in others if abs(v.width - W) < 0.03 * W or abs(v.height - Hh) < 0.03 * Hh]
        if side:
            v = side[0]
            thick = round(v.height if abs(v.width - W) < 0.03 * W else v.width, 2)
            # prefer a stated dimension that matches
            m = [c.length for c in cs if c.length and abs(c.length - thick) < 0.04 * thick + 0.3]
            if m:
                thick = m[0]
            review.append(ReviewItem("thickness", f"thickness {thick} taken from side view"))
        else:
            thick = 10.0
            review.append(ReviewItem("thickness", "part thickness not found: set part_z", "block"))

    # --- holes: group concentric circles ---
    circles = [c for c in top.circles]
    groups = []
    for c in sorted(circles, key=lambda c: c.r):
        for gpp in groups:
            if math.hypot(gpp[0].cx - c.cx, gpp[0].cy - c.cy) < max(0.05, 0.02 * c.r) * (10 if is_scan else 1):
                gpp.append(c)
                break
        else:
            groups.append([c])

    hole_calls = [c for c in cs if c.dia or c.thread]
    holes, pockets = [], []
    def snap_xy(v):
        if is_scan and abs(v * 2 - round(v * 2)) < 0.2:      # drawings use round coordinates
            return round(v * 2) / 2
        return v

    for gpp in groups:
        x, y = snap_xy(gpp[0].cx - ox), snap_xy(gpp[0].cy - oy)
        if not outline_poly.buffer(0.1).contains(Point(gpp[0].cx, gpp[0].cy)):
            continue
        full = [c for c in gpp if not c.partial]
        part = [c for c in gpp if c.partial]
        dia = 2 * (full[0].r if full else gpp[0].r)
        # which callout describes this hole: same size first (handles "4X"), then nearest
        def matches(c):
            if c.thread:
                major, pitch = c.thread
                return (part and abs(2 * part[0].r - major) < 0.05 * major + 0.2) or abs(dia - (major - pitch)) < 0.06 * major + 0.2
            return abs(c.dia - dia) < (0.04 * dia + 0.3 if is_scan else 0.02)
        cands = [c for c in hole_calls if matches(c)]
        call, _ = nearest(cands, gpp[0].cx, gpp[0].cy) if cands else (None, None)
        if call is None and is_scan and dia < 4.0:
            review.append(ReviewItem("hole", f"ignored a Ø{dia:.1f} circle at X{x:.1f} Y{y:.1f} with no callout (likely a symbol)"))
            continue
        if call is None:
            review.append(ReviewItem("hole", f"hole Ø{dia:.2f} at X{x:.2f} Y{y:.2f} has no callout; assumed drilled THRU"))
        if call and call.thread:
            major, pitch = call.thread
            depth = call.depth or thick
            holes.append(Hole(round(x, 3), round(y, 3), major, depth, "tap", pitch, through=not call.depth or call.through))
            continue
        reamed = bool(call and (call.ream or (call.tol and call.tol[0] - call.tol[1] <= 0.03)))
        if call:
            # a reamer sets the size itself; drilled holes are programmed to mid-tolerance
            dia = call.dia if reamed else call.mid_tol_value(call.dia)
        if len(full) >= 2:
            review.append(ReviewItem("hole", f"concentric circles at X{x:.1f} Y{y:.1f}: counterbore/spotface; only the through hole is programmed"))
        through = (call is None) or call.through or not call.depth
        depth = thick if through else call.depth
        if dia > MAX_DRILL:
            pockets.append(Pocket("circle", depth=depth + (0.5 if through else 0), cx=round(x, 3), cy=round(y, 3), r=round(dia / 2, 3)))
            continue
        kind = "ream" if reamed else "drill"
        holes.append(Hole(round(x, 3), round(y, 3), round(dia, 3), depth, kind, through=through))

    # --- pockets: closed faces strictly inside the outline ---
    for f in faces[1:]:
        if not f.within(outline_poly.buffer(0.01)) or f.area > 0.9 * outline_poly.area:
            continue
        # faces created by circles are not in this list (circles are not segments)
        ri = _rect_info(f)
        c, dist = nearest(cs, f.centroid.x, f.centroid.y, lambda c: c.depth is not None and not c.dia and not c.thread)
        depth = c.depth if c and dist < 2.0 * math.sqrt(f.area) else None
        if depth is None:
            review.append(ReviewItem("pocket", f"pocket at X{f.centroid.x - ox:.1f} Y{f.centroid.y - oy:.1f} has no depth callout", "block"))
            depth = 1.0
        if ri:
            cx, cy, w, h, r = ri
            if is_scan:
                cx, cy, w, h = snap_v(cx - ox) + ox, snap_v(cy - oy) + oy, snap_v(w), snap_v(h)
                r = snap_v(r, 0.4)
            if r < 0.5:
                review.append(ReviewItem("pocket", f"pocket {w:.1f}x{h:.1f} has sharp internal corners: not millable, will leave tool radius"))
            pockets.append(Pocket("rect", depth=depth, cx=round(cx - ox, 3), cy=round(cy - oy, 3), w=round(w, 3), h=round(h, 3), corner_r=r))
        else:
            pts = [[round(px - ox, 3), round(py - oy, 3)] for px, py in f.exterior.coords]
            pockets.append(Pocket("poly", depth=depth, points=pts))

    material = g["material"]
    if not material:
        review.append(ReviewItem("material", "material not found on drawing; cutting data needs it", "block"))
        material = "MS"
    outline = [[round(px - ox, 3), round(py - oy, 3)] for px, py in outline_poly.exterior.coords]
    review.append(ReviewItem("stock", "stock assumed 2 mm oversize per side and 2 mm on top; edit stock_* if blanks are pre-sized"))
    if is_scan:
        review.append(ReviewItem("geometry", "positions read from a scan: verify every hole X/Y against the drawing", "warn"))
    return MillSpec(part_name=g["part_name"] or "PART", material=material,
                    stock_x=round(W + 4, 2), stock_y=round(Hh + 4, 2), stock_z=round(thick + 2, 2), part_z=thick,
                    outline=outline, holes=holes, pockets=pockets, review=review)
