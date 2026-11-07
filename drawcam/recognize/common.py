"""Shared recognition helpers: views, callouts, scale calibration, machine-type features."""
from __future__ import annotations

import math
import os

import numpy as np

from ..ingest.dim_parser import parse, parse_globals
from ..ingest.drawing import Drawing
from ..spec import ReviewItem


# ---------------------------------------------------------------- views ----
class View:
    def __init__(self):
        self.segs, self.circles = [], []

    def bbox(self):
        xs, ys = [], []
        for s in self.segs:
            if s.kind == "visible":
                xs += [s.x1, s.x2]
                ys += [s.y1, s.y2]
        for c in self.circles:
            xs += [c.cx - c.r, c.cx + c.r]
            ys += [c.cy - c.r, c.cy + c.r]
        return (min(xs), min(ys), max(xs), max(ys)) if xs else (0, 0, 0, 0)

    @property
    def width(self):
        b = self.bbox()
        return b[2] - b[0]

    @property
    def height(self):
        b = self.bbox()
        return b[3] - b[1]


def split_views(d: Drawing, gap: float = 4.0) -> list:
    """Group visible geometry into views (connected clusters, bounding-box proximity).
    Centre / hidden lines are attached to whichever view they overlap most."""
    items = [("s", s) for s in d.segs if s.kind == "visible"] + [("c", c) for c in d.circles if c.kind != "thin" or c.partial]

    def box(it):
        k, o = it
        if k == "s":
            return (min(o.x1, o.x2), min(o.y1, o.y2), max(o.x1, o.x2), max(o.y1, o.y2))
        return (o.cx - o.r, o.cy - o.r, o.cx + o.r, o.cy + o.r)

    boxes = [box(i) for i in items]
    n = len(items)
    parent = list(range(n))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    order = sorted(range(n), key=lambda i: boxes[i][0])
    for ii, i in enumerate(order):                      # sweep on x for speed
        bi = boxes[i]
        for j in order[ii + 1:]:
            bj = boxes[j]
            if bj[0] > bi[2] + gap:
                break
            if bj[1] <= bi[3] + gap and bj[3] >= bi[1] - gap:
                parent[find(i)] = find(j)
    # Anything lying inside another cluster's box belongs to that view (holes
    # inside an outline do not touch the outline lines).
    while True:
        roots = {}
        for i in range(n):
            r = find(i)
            b = boxes[i]
            if r in roots:
                o = roots[r]
                roots[r] = (min(o[0], b[0]), min(o[1], b[1]), max(o[2], b[2]), max(o[3], b[3]))
            else:
                roots[r] = b
        merged = False
        keys = list(roots)
        for a in keys:
            for b in keys:
                if a == b or find(a) == find(b):
                    continue
                A, B = roots[a], roots[b]
                if A[0] >= B[0] - 0.01 and A[1] >= B[1] - 0.01 and A[2] <= B[2] + 0.01 and A[3] <= B[3] + 0.01:
                    parent[find(a)] = find(b)
                    merged = True
        if not merged:
            break
    groups = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(items[i])
    views = []
    for g in groups.values():
        v = View()
        for k, o in g:
            (v.segs if k == "s" else v.circles).append(o)
        if v.width > 2 or v.height > 2:
            views.append(v)
    # attach centre/hidden lines
    for s in d.segs:
        if s.kind not in ("center", "hidden"):
            continue
        best, ov = None, 0
        for v in views:
            b = v.bbox()
            lo, hi = max(b[0] - gap, min(s.x1, s.x2)), min(b[2] + gap, max(s.x1, s.x2))
            ylo, yhi = max(b[1] - gap, min(s.y1, s.y2)), min(b[3] + gap, max(s.y1, s.y2))
            o = max(hi - lo, 0) + max(yhi - ylo, 0) if (hi >= lo and yhi >= ylo) else 0
            if o > ov:
                best, ov = v, o
        if best:
            best.segs.append(s)
    views.sort(key=lambda v: v.width * v.height, reverse=True)
    return views


# ------------------------------------------------------------- callouts ----
def callouts(d: Drawing):
    ocr = d.source == "scan"
    return [parse(t.text, t.x, t.y, t.conf, ocr=ocr) for t in d.texts]


def globals_(d: Drawing):
    return parse_globals(d.texts)


# ---------------------------------------------------------- calibration ----
def side_view(views):
    """Lathe side view: has a long horizontal centre line and horizontal outline lines."""
    def has_axis(v):
        return any(s.kind == "center" and s.is_horizontal(0.01) and s.length > 0.6 * v.width for s in v.segs)
    cands = [v for v in views if has_axis(v)] or views
    return max(cands, key=lambda v: (sum(1 for s in v.segs if s.kind == "visible" and s.is_horizontal(0.01)), v.width))


def top_view(views):
    return max(views, key=lambda v: (len(v.circles), v.width * v.height))


def calibrate(d: Drawing, review: list, machine: str) -> None:
    """Convert paper/pixel mm to model mm.
    DXF is already in model units. PDF/scan: estimate the factor from the
    largest dimensions, cross-check against the title-block SCALE."""
    if d.source == "dxf":
        return
    g = globals_(d)
    views = split_views(d)
    if not views:
        review.append(ReviewItem("geometry", "no outline geometry found", "block"))
        return
    cs = callouts(d)
    lengths = sorted([c.length for c in cs if c.length], reverse=True)
    dias = sorted([c.dia for c in cs if c.dia and not c.thread] + [c.thread[0] for c in cs if c.thread], reverse=True)

    ests = []
    if machine == "lathe":
        main = side_view(views)
        if dias and main.height:
            ests.append(dias[0] / main.height)
        if lengths and main.width:
            ests.append(lengths[0] / main.width)
    else:
        main = top_view(views)
        if lengths:
            big = max(main.width, main.height)
            ests.append(lengths[0] / big)
    scale = g["scale"]
    est = None
    if ests:
        est = ests[0]
        if len(ests) > 1 and abs(ests[0] - ests[1]) / ests[0] > 0.02:
            # disagreement usually means a dimension was mis-read or missed; trust the
            # estimate that agrees with the title block, else the overall length
            pick = None
            if scale:
                pick = next((e for e in ests if abs(e - 1 / scale) * scale < 0.02), None)
            est = pick or ests[-1]
            review.append(ReviewItem("scale", "overall length and largest diameter imply different scales; check the profile", "warn"))

    if scale:
        model_per_paper = 1.0 / scale
        if est and abs(est - model_per_paper) / model_per_paper > 0.02:
            review.append(ReviewItem("scale", f"title block SCALE disagrees with dimensions (fits {1/est:.3f}:1); using dimensions", "warn"))
            model_per_paper = est
    elif est:
        model_per_paper = est
        if d.source == "pdf":
            review.append(ReviewItem("scale", f"no SCALE in title block; calibrated from dimensions (x{est:.4f})", "warn"))
    else:
        model_per_paper = 1.0
        review.append(ReviewItem("scale", "could not determine drawing scale; sizes may be wrong", "block"))
    d.transform(model_per_paper)
    d.scale_known = True


# ------------------------------------------------------ machine features ----
FEATURES = ["has_h_center", "symmetry", "n_circles_main", "n_circles_total", "aspect", "dia_text_ratio",
            "thread_text", "n_views", "concentric_small_view", "hole_count_text"]


def machine_features(d: Drawing) -> dict:
    views = split_views(d)
    f = dict.fromkeys(FEATURES, 0.0)
    if not views:
        return f

    def axis_line(v):
        c = [s for s in v.segs if s.kind == "center" and s.is_horizontal(0.01) and s.length > 0.7 * v.width]
        return max(c, key=lambda s: s.length) if c else None

    def sym(v, axis):
        """Fraction of outline (sampled every ~1 mm) that has a mirror image across the axis.
        Point sampling makes it robust to fragmented scan lines."""
        pts = []
        for s in v.segs:
            if s.kind != "visible":
                continue
            n = max(2, int(s.length))
            for k in range(n + 1):
                t = k / n
                pts.append((s.x1 + t * (s.x2 - s.x1), s.y1 + t * (s.y2 - s.y1)))
        if len(pts) < 4:
            return 0.0
        P = np.array(pts)
        M = P.copy()
        M[:, 1] = 2 * axis - M[:, 1]
        tol = 0.01 * max(v.width, v.height) + 0.6
        hit = 0
        for i in range(0, len(M), 64):
            dd = np.sqrt(((M[i:i + 64, None, :] - P[None, :, :]) ** 2).sum(-1)).min(1)
            hit += int((dd < tol).sum())
        return hit / len(M)

    with_axis = [v for v in views if axis_line(v) is not None and v.width > v.height * 0.4]
    def score(v):
        ax_ = axis_line(v)
        bb = v.bbox()
        return sym(v, ax_.y1 if ax_ else (bb[1] + bb[3]) / 2)
    main = max(with_axis or views, key=lambda v: (round(score(v), 2), v.width * v.height))
    ax = axis_line(main)
    f["has_h_center"] = 1.0 if ax else 0.0
    f["symmetry"] = score(main)
    f["n_views"] = len(views)
    f["aspect"] = main.width / max(main.height, 1e-6)
    f["n_circles_main"] = len(main.circles)
    f["n_circles_total"] = len(d.circles)
    cs = callouts(d)
    nd = sum(1 for c in cs if c.dia)
    nl = sum(1 for c in cs if c.length)
    f["dia_text_ratio"] = nd / max(nd + nl, 1)
    f["thread_text"] = float(sum(1 for c in cs if c.thread))
    f["hole_count_text"] = float(sum(1 for c in cs if c.count > 1 or c.through))
    for v in views:
        if v is main or len(v.circles) < 2:
            continue
        cx = [c.cx for c in v.circles]
        cy = [c.cy for c in v.circles]
        if np.ptp(cx) < 0.5 and np.ptp(cy) < 0.5:
            f["concentric_small_view"] = 1.0
    return f


MODEL_PATH = os.path.join(os.path.dirname(__file__), "..", "ml", "machine_classifier.joblib")


def classify_machine(d: Drawing):
    """Returns (machine, confidence, how). Trained model if present, else rules."""
    f = machine_features(d)
    if os.path.exists(MODEL_PATH):
        import joblib
        model = joblib.load(MODEL_PATH)
        x = np.array([[f[k] for k in FEATURES]])
        p = model.predict_proba(x)[0]
        i = int(np.argmax(p))
        return model.classes_[i], float(p[i]), "ml"
    return rule_classify(f)


def rule_classify(f: dict):
    score = 0.45 * f["has_h_center"] + 0.35 * (f["symmetry"] > 0.6) + 0.2 * f["concentric_small_view"] - 0.2 * (f["n_circles_main"] > 0)
    if score >= 0.6:
        return "lathe", min(0.95, 0.5 + score / 2), "rules"
    if score <= 0.2:
        return "mill", min(0.95, 0.9 - score), "rules"
    return ("lathe" if score > 0.4 else "mill"), 0.6, "rules"


# ---------------------------------------------------------- utilities ----
def snap(value: float, candidates, rel=0.03, abs_=0.3):
    best = None
    for c in candidates:
        if abs(c - value) <= max(abs_, rel * c):
            if best is None or abs(c - value) < abs(best - value):
                best = c
    return best


def nearest(callouts_, x, y, pred=lambda c: True):
    cs = [c for c in callouts_ if pred(c)]
    if not cs:
        return None, math.inf
    c = min(cs, key=lambda c: math.hypot(c.x - x, c.y - y))
    return c, math.hypot(c.x - x, c.y - y)
