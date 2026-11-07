"""Drawing -> TurnSpec. Reads the side view of a turned part: finds the axis,
traces the upper outline (the half-profile), then attaches threads, grooves,
bores and tolerances from the dimension text."""
from __future__ import annotations

from ..ingest.drawing import Drawing
from ..spec import TurnSpec, Thread, Groove, Bore, ReviewItem
from .common import split_views, side_view, callouts, globals_, snap, nearest

BAR_SIZES = [6, 8, 10, 12, 16, 20, 22, 25, 28, 30, 32, 36, 40, 45, 50, 55, 60, 65, 70, 75, 80, 90,
             100, 110, 120, 130, 140, 150, 160, 180, 200, 220, 250]


def _envelope(segs, axis, x):
    best = None
    for s in segs:
        lo, hi = min(s.x1, s.x2), max(s.x1, s.x2)
        if hi - lo < 1e-6 or not (lo - 1e-9 <= x <= hi + 1e-9):
            continue
        t = (x - s.x1) / (s.x2 - s.x1)
        y = s.y1 + t * (s.y2 - s.y1) - axis
        if y > 0 and (best is None or y > best):
            best = y
    return best


def _bridge_gaps(segs, ytol=0.4, gap=10.0):
    """Scans break long lines where text was erased; rejoin horizontal pieces on one
    line, but never across a vertical edge (that would erase a real groove or step)."""
    from ..ingest.drawing import Seg
    hor = sorted([s for s in segs if s.is_horizontal(0.01)], key=lambda s: (s.y1 + s.y2) / 2)
    rest = [s for s in segs if not s.is_horizontal(0.01)]
    verts = [s for s in rest if s.is_vertical(0.02)]
    rows, out = [], []
    for s in hor:
        y = (s.y1 + s.y2) / 2
        if rows and abs(rows[-1][0] - y) < ytol:
            rows[-1][1].append(s)
        else:
            rows.append([y, [s]])
    for y, row in rows:
        row.sort(key=lambda s: min(s.x1, s.x2))
        cur = None
        for s in row:
            a, b = min(s.x1, s.x2), max(s.x1, s.x2)
            if cur and a - cur.x2 < gap and not any(
                    cur.x2 + 0.3 < v.x1 < a - 0.3 and min(v.y1, v.y2) - 0.5 <= y <= max(v.y1, v.y2) + 0.5 for v in verts):
                cur.x2 = max(cur.x2, b)
                continue
            cur = Seg(a, y, b, y, "visible")
            out.append(cur)
    return out + rest


def _clean(points, tol=1e-3):
    out = []
    for p in points:
        if out and abs(out[-1][0] - p[0]) < tol and abs(out[-1][1] - p[1]) < tol:
            continue
        out.append(p)
    # drop collinear middle points
    i = 1
    while i < len(out) - 1:
        (z0, d0), (z1, d1), (z2, d2) = out[i - 1], out[i], out[i + 1]
        if abs((z1 - z0) * (d2 - d0) - (z2 - z0) * (d1 - d0)) < tol * max(1, abs(z2 - z0)):
            out.pop(i)
        else:
            i += 1
    return out


def recognize_turning(d: Drawing) -> TurnSpec:
    review = []
    views = split_views(d)
    main = side_view(views)
    b = main.bbox()
    cen = [s for s in main.segs if s.kind == "center" and s.is_horizontal(0.01)]
    if cen:
        axis = max(cen, key=lambda s: s.length).y1
    else:
        axis = (b[1] + b[3]) / 2
        review.append(ReviewItem("axis", "no centre line found; assumed axis at mid-height of the side view"))

    vis = [s for s in main.segs if s.kind == "visible" and max(s.y1, s.y2) > axis + 1e-3]
    if d.source == "scan":
        vis = _bridge_gaps(vis)
    xmax = b[2]
    xs = sorted({round(v, 4) for s in vis for v in (s.x1, s.x2)})
    eps = 1e-4
    pts = []
    for xa, xb in zip(reversed(xs[1:]), reversed(xs[:-1])):      # right -> left
        ra = _envelope(vis, axis, xa - eps)
        rb = _envelope(vis, axis, xb + eps)
        if ra is None or rb is None:
            continue
        pts.append([round(xa - xmax, 4), round(2 * ra, 4)])
        pts.append([round(xb - xmax, 4), round(2 * rb, 4)])
    profile = _clean(pts)
    if len(profile) < 2:
        raise ValueError("could not trace a turned profile from the side view")

    cs = callouts(d)
    g = globals_(d)
    is_scan = d.source == "scan"

    # --- snap diameters to stated dimensions, program to mid-tolerance ---
    dia_calls = [c for c in cs if c.dia and not c.thread]
    thr_calls = [c for c in cs if c.thread]
    stated = sorted({c.dia for c in dia_calls} | {c.thread[0] for c in thr_calls})
    for p in profile:
        s = snap(p[1], stated, rel=0.04 if is_scan else 0.005, abs_=0.6 if is_scan else 0.02)
        if s is not None:
            c = next((c for c in dia_calls if c.dia == s), None)
            p[1] = round(c.mid_tol_value(s) if c else s, 3)
            if c and c.fit and not c.tol:
                lim = c.limits(s)
                if lim:
                    review.append(ReviewItem("tolerance", f"Ø{s} {c.fit} = {lim[0]:+.3f}/{lim[1]:+.3f}; programmed to mid-limit {p[1]}"))
                else:
                    review.append(ReviewItem("tolerance", f"Ø{s} {c.fit}: fit not in table, programmed to nominal"))
        elif is_scan:
            raw_d = p[1]
            if abs(p[1] * 2 - round(p[1] * 2)) < 0.5:
                p[1] = round(p[1] * 2) / 2
            review.append(ReviewItem("profile", f"diameter read as {raw_d:.2f}, set to {p[1]:g}: no matching dimension text"))
    if is_scan:
        for p in profile:
            p[0] = round(p[0] * 2) / 2
        review.append(ReviewItem("profile", "step positions read from a scan and rounded to 0.5 mm: check every Z against the drawing"))
    profile = _clean(profile)

    part_length = round(-profile[-1][0], 3)
    lens = [c.length for c in cs if c.length]
    if lens:
        L = max(lens)
        if abs(L - part_length) > max(0.05, 0.01 * L):
            review.append(ReviewItem("length", f"traced length {part_length} vs largest dimension {L}; using {L}" if is_scan else
                                     f"traced length {part_length} differs from dimension {L}", "warn"))
            if is_scan:
                k = L / part_length
                for p in profile:
                    p[0] = round(p[0] * k, 2)
                part_length = L

    # --- threads ---
    threads = []
    for c in thr_calls:
        major, pitch = c.thread
        seg = None
        for (z0, d0), (z1, d1) in zip(profile, profile[1:]):
            if abs(d0 - d1) < 1e-6 and abs(d0 - major) <= max(0.3, 0.02 * major) and z0 - z1 > pitch:
                seg = (z0, z1)
                break
        if seg is None:
            review.append(ReviewItem("thread", f"M{major}x{pitch} stated but no matching cylinder found", "block"))
            continue
        threads.append(Thread(z_start=seg[0], z_end=seg[1], major_dia=major, pitch=pitch))
        od = round(major - 0.1 * pitch, 3)            # turn thread OD slightly under major
        for p in profile:
            if abs(p[1] - major) < 1e-6 and seg[1] - 1e-6 <= p[0] <= seg[0] + 1e-6:
                p[1] = od

    # --- grooves: short dips lower than both neighbours ---
    # Pattern in the point list: (za,D1) (za,dg) (zb,dg) (zb,D2) with dg below both
    # neighbours and za-zb <= 6 mm. Roughing (G71 type I) cannot dip, so the groove
    # is filled in the profile and cut afterwards with a grooving tool.
    grooves = []
    i = 0
    while i + 3 < len(profile):
        (za, d1), (za2, dg), (zb, dg2), (zb2, d2) = profile[i:i + 4]
        if (abs(za - za2) < 1e-6 and abs(zb - zb2) < 1e-6 and abs(dg - dg2) < 1e-6
                and dg < d1 - 1e-6 and dg < d2 - 1e-6 and 0 < za - zb <= 6.0):
            fill = min(d1, d2)
            grooves.append(Groove(z=round(zb, 3), width=round(za - zb, 3), bottom_dia=dg))
            profile[i + 1][1] = fill
            profile[i + 2][1] = fill
            i += 3
        else:
            i += 1
    profile = _clean(profile)

    # --- bore ---
    bore = None
    bc = next((c for c in dia_calls if ("DRILL" in c.keywords or c.depth or c.through) and c.dia < min(p[1] for p in profile)), None)
    if bc:
        bore = Bore(dia=bc.mid_tol_value(bc.dia), depth=bc.depth or part_length, through=bc.through or not bc.depth)
        if bore.through:
            bore.depth = part_length
    else:
        hid = [s for s in main.segs if s.kind == "hidden" and s.is_horizontal(0.01) and s.y1 > axis + 0.1]
        if hid:
            r = min(s.y1 for s in hid) - axis
            right = [s for s in hid if abs(s.y1 - axis - r) < 0.05]
            depth = xmax - min(min(s.x1, s.x2) for s in right)
            bore = Bore(dia=round(2 * r, 3), depth=round(depth, 3), through=depth >= part_length - 0.01)
            review.append(ReviewItem("bore", f"bore Ø{bore.dia} x {bore.depth} taken from hidden lines (no callout)"))

    max_dia = max(p[1] for p in profile)
    stock_dia = next((s for s in BAR_SIZES if s >= max_dia + 2.0), round(max_dia + 5))
    chamfers = [c.chamfer for c in cs if c.chamfer]
    material = g["material"]
    if not material:
        review.append(ReviewItem("material", "material not found on drawing; cutting data needs it", "block"))
        material = "MS"

    # non-monotonic profile = needs a second setup
    peak = max(range(len(profile)), key=lambda i: (profile[i][1], -i))
    if any(profile[j][1] < profile[j - 1][1] - 1e-6 for j in range(peak + 1, len(profile))):
        review.append(ReviewItem("setup", "profile gets smaller again past the largest diameter: part needs two setups (program 2 is generated, flip the part)"))

    return TurnSpec(
        part_name=g["part_name"] or "PART", material=material, part_length=part_length, max_dia=max_dia,
        stock_dia=stock_dia, stock_length=round(part_length + 4.0, 1), profile=profile,
        threads=threads, grooves=grooves, bore=bore, chamfer=max(chamfers) if chamfers else 0.5,
        review=review)
