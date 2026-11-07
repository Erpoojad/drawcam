"""Reads back the generated G-code (not the planner's intent) and checks it:
rapids into material, missing feeds, RPM limits, travel, depth below the part.
Also produces a backplot PNG and a cycle-time estimate for quoting."""
from __future__ import annotations

import math
import re

WORD = re.compile(r"([A-Z])\s*(-?\d*\.?\d*)")
RAPID_MM_MIN = 12000.0


def _words(line):
    line = re.sub(r"\(.*?\)", "", line).strip()
    return [(a, float(b)) for a, b in WORD.findall(line) if b not in ("", "-", ".")], line


class Issue:
    def __init__(self, line_no, severity, msg):
        self.line, self.severity, self.msg = line_no, severity, msg

    def __repr__(self):
        return f"{self.severity.upper()} line {self.line}: {self.msg}"


# ===================================================================== MILL
def sim_mill(code: str, spec, cfg):
    moves, issues = [], []
    x = y = 0.0
    z = 100.0
    mode = 0
    feed = None
    s = 0
    cyc = None
    cyc_z = cyc_r = None
    init_z = z
    ztop = spec.stock_z - spec.part_z          # material top; drops once the face mill has run
    face_t = cfg["mill"]["tools"]["face"]["t"]
    tool = None
    inc = False
    t_feed = t_rapid = 0.0
    max_rpm = cfg["mill"]["max_rpm"]
    tr = cfg["mill"]["travel"]
    for n, raw in enumerate(code.splitlines(), 1):
        w, line = _words(raw)
        if not w:
            continue
        d = {}
        gs = []
        for a, v in w:
            if a == "G":
                gs.append(int(round(v)))
            else:
                d[a] = v
        if "M" in d and int(d["M"]) == 6 and "T" in d:
            tool = int(d["T"])
        if 91 in gs:
            inc = True
        if 90 in gs:
            inc = False
        if 28 in gs:
            if "Z" in d:
                moves.append(("rapid", (x, y, z), (x, y, 100.0)))
                z = 100.0
            continue
        if inc:
            continue
        if "S" in d:
            s = d["S"]
            if s > max_rpm:
                issues.append(Issue(n, "error", f"S{s:g} above machine max {max_rpm}"))
        if "F" in d:
            feed = d["F"]
        for gg in gs:
            if gg in (0, 1, 2, 3):
                mode, cyc = gg, None
            if gg in (81, 83, 84, 85):
                cyc = gg
                cyc_z, cyc_r = d.get("Z"), d.get("R")
            if gg == 80:
                cyc = None
            if gg == 43 and "Z" in d:
                pass
        if cyc and ("X" in d or "Y" in d):
            if "Z" in d and any(g_ in (81, 83, 84, 85) for g_ in gs):
                cyc_z = d["Z"]
            if "R" in d:
                cyc_r = d["R"]
            init_z = z
            nx, ny = d.get("X", x), d.get("Y", y)
            moves.append(("rapid", (x, y, z), (nx, ny, z)))
            t_rapid += math.dist((x, y), (nx, ny)) / RAPID_MM_MIN
            x, y = nx, ny
            if feed is None:
                issues.append(Issue(n, "error", "canned cycle without feed"))
            if cyc_r is not None and cyc_r < ztop + 0.5 and cyc_r > -0.01:
                issues.append(Issue(n, "warn", f"R plane {cyc_r:g} close to stock top {ztop:g}"))
            moves.append(("feed", (x, y, cyc_r), (x, y, cyc_z)))
            depth = (cyc_r - cyc_z)
            mult = 2 if cyc in (84, 85) else (1.5 if cyc == 83 else 1)
            t_feed += depth * mult / max(feed or 1, 1)
            continue
        if not any(k in d for k in "XYZ") or cyc:
            continue
        nx, ny, nz = d.get("X", x), d.get("Y", y), d.get("Z", z)
        if mode == 0:
            # rapid into stock: XY travel at or below stock top, or plunge below it
            if nz < ztop + 0.5 - 1e-6 and (nx != x or ny != y) and z < ztop + 0.5:
                inside = (-1e-6 - (spec.stock_x - _w(spec)) / 2 <= nx <= _w(spec) + (spec.stock_x - _w(spec)) / 2)
                issues.append(Issue(n, "error" if inside else "warn", f"rapid XY move at Z{nz:g} (stock top Z{ztop:g})"))
            if nz < ztop + 0.5 - 1e-6 and nz < z and (nx == x and ny == y):
                issues.append(Issue(n, "error", f"rapid plunge to Z{nz:g} below stock top"))
            moves.append(("rapid", (x, y, z), (nx, ny, nz)))
            t_rapid += math.dist((x, y, z), (nx, ny, nz)) / RAPID_MM_MIN
        else:
            if feed is None:
                issues.append(Issue(n, "error", "feed move without F"))
            if mode in (2, 3):
                i, j = d.get("I", 0.0), d.get("J", 0.0)
                cx, cy = x + i, y + j
                r = math.hypot(i, j)
                a0 = math.atan2(y - cy, x - cx)
                a1 = math.atan2(ny - cy, nx - cx)
                sweep = a1 - a0
                if mode == 3:
                    sweep = sweep % (2 * math.pi) or 2 * math.pi
                else:
                    sweep = -((-sweep) % (2 * math.pi) or 2 * math.pi)
                steps = max(8, int(abs(sweep) * r / 1.0))
                px, py, pz = x, y, z
                for k in range(1, steps + 1):
                    a = a0 + sweep * k / steps
                    qx, qy, qz = cx + r * math.cos(a), cy + r * math.sin(a), z + (nz - z) * k / steps
                    moves.append(("feed", (px, py, pz), (qx, qy, qz)))
                    px, py, pz = qx, qy, qz
                L = math.hypot(abs(sweep) * r, nz - z)
            else:
                moves.append(("feed", (x, y, z), (nx, ny, nz)))
                L = math.dist((x, y, z), (nx, ny, nz))
            t_feed += L / max(feed or 1, 1)
            if tool == face_t and nz < ztop:
                ztop = max(nz, 0.0)
            if nz < -(spec.part_z + 1.5):
                issues.append(Issue(n, "warn", f"cutting at Z{nz:g}, more than 1.5 mm below the part: check table/parallels"))
        x, y, z = nx, ny, nz
        if not (-tr[0] <= x <= tr[0] and -tr[1] <= y <= tr[1]):
            issues.append(Issue(n, "error", f"X{x:g} Y{y:g} outside travel"))
    tool_changes = code.count("M06")
    minutes = t_feed + t_rapid + tool_changes * 0.15
    return moves, issues, minutes


def _w(spec):
    xs = [p[0] for p in spec.outline] or [0]
    return max(xs) - min(xs)


# ==================================================================== LATHE
def _profile_dia(profile, z):
    best = 0.0
    for (za, da), (zb, db) in zip(profile, profile[1:]):
        lo, hi = min(za, zb), max(za, zb)
        if lo - 1e-6 <= z <= hi + 1e-6:
            if abs(za - zb) < 1e-9:
                best = max(best, da, db)
            else:
                best = max(best, da + (db - da) * (z - za) / (zb - za))
    return best


def sim_lathe(code: str, spec, cfg, flipped=False):
    moves, issues = [], []
    lines = code.splitlines()
    nidx = {}
    for i, raw in enumerate(lines):
        m = re.match(r"\s*N(\d+)", raw)
        if m:
            nidx[int(m.group(1))] = i
    x, z = 300.0, 200.0
    mode = 0
    feed = None
    css = False
    s_val = 0
    smax = cfg["lathe"]["max_rpm"]
    t = 0.0
    skip_until = -1
    pending71 = None
    prof = spec.profile if not flipped else [[-spec.part_length - zz, dd] for zz, dd in reversed(spec.profile)]
    bore_d = spec.bore.dia if spec.bore else 0.0
    shaped = False                     # after OD roughing, material = finished profile

    def spindle(dia):
        if css:
            return min(smax, 1000 * s_val / (math.pi * max(dia, 1)))
        return s_val

    def feed_time(L, dia):
        return L / max((feed or 0.1) * spindle(dia), 1)

    def contour(p, q):
        pts = []
        cx, cz = x, z
        for k in range(nidx[p], nidx[q] + 1):
            w, _ = _words(lines[k])
            d = {a: v for a, v in w if a in "XZ"}
            cx, cz = d.get("X", cx), d.get("Z", cz)
            pts.append((cx, cz))
        return pts

    for n, raw in enumerate(lines, 1):
        if n - 1 <= skip_until:
            continue
        w, line = _words(raw)
        if not w:
            continue
        d, gs = {}, []
        for a, v in w:
            if a == "G":
                gs.append(int(round(v)))
            elif a != "N":
                d[a] = v
        if 28 in gs:
            moves.append(("rapid", (x, z), (300.0, z if "U" in d else 200.0)))
            x = 300.0 if "U" in d else x
            z = 200.0 if "W" in d else z
            continue
        if 50 in gs and "S" in d:
            smax = min(smax, d["S"])
            continue
        if 96 in gs:
            css = True
        if 97 in gs:
            css = False
        if "S" in d:
            s_val = d["S"]
            if not css and s_val > cfg["lathe"]["max_rpm"]:
                issues.append(Issue(n, "error", f"S{s_val:g} above machine max"))
        if "F" in d and not any(g_ in (71, 70, 74, 75, 76) for g_ in gs):
            feed = d["F"]
        if 71 in gs:
            if "P" in d:
                feed = d.get("F", feed)
                pts = contour(int(d["P"]), int(d["Q"]))
                internal = d.get("U", 0) < 0
                skip_until = nidx[int(d["Q"])]
                # roughing passes: horizontal cuts at each depth level
                doc = pending71 or 2.0
                start_x, start_z = x, z
                if not internal:
                    lv = start_x - 2 * doc
                    while lv > min(p[0] for p in pts) + 0.1:
                        zend = min((pz for px, pz in pts if px <= lv + 1e-6), default=start_z)
                        zcut = _cut_end(pts, lv)
                        moves.append(("feed", (lv, start_z), (lv, zcut)))
                        t += feed_time(start_z - zcut, lv)
                        lv -= 2 * doc
                    shaped = True
                else:
                    lv = start_x + 2 * doc
                    while lv < max(p[0] for p in pts) - 0.1:
                        zcut = _cut_end(pts, lv, internal=True)
                        moves.append(("feed", (lv, start_z), (lv, zcut)))
                        t += feed_time(start_z - zcut, lv)
                        lv += 2 * doc
                for a, b in zip(pts, pts[1:]):
                    moves.append(("feed", a, b))
                    t += feed_time(math.dist(a, b), a[0])
            else:
                pending71 = d.get("U", 2.0)
            continue
        if 70 in gs:
            f70 = d.get("F", feed)
            pts = contour(int(d["P"]), int(d["Q"]))
            for a, b in zip(pts, pts[1:]):
                moves.append(("finish", a, b))
                t += math.dist(a, b) / max(f70 * spindle(a[0]), 1)
            continue
        if 74 in gs:
            if "Z" in d:
                moves.append(("feed", (x, z), (x, d["Z"])))
                t += abs(z - d["Z"]) * 1.4 / max(d.get("F", 0.1) * s_val, 1)
            continue
        if 75 in gs:
            if "X" in d:
                f75 = d.get("F", 0.05)
                z_to = d.get("Z", z)
                q = d.get("Q", 1e9) / 1000
                zz = z
                while True:
                    moves.append(("feed", (x, zz), (d["X"], zz)))
                    t += abs(x - d["X"]) / 2 * 1.3 / max(f75 * s_val, 1)
                    if zz <= z_to + 1e-6:
                        break
                    zz = max(z_to, zz - q)
            continue
        if 76 in gs:
            if "X" in d:
                h = d.get("P", 1000) / 1000
                q = d.get("Q", 200) / 1000
                passes = max(4, int((h / q) ** 2)) + 2
                moves.append(("thread", (d["X"], z), (d["X"], d["Z"])))
                t += passes * (abs(z - d["Z"]) + 6) / max(d.get("F", 1) * s_val, 1)
            continue
        for gg in gs:
            if gg in (0, 1, 2, 3):
                mode = gg
        if not any(k in d for k in "XZ"):
            continue
        nx, nz = d.get("X", x), d.get("Z", z)
        if mode == 0:
            # rapid target inside material?
            mat = _profile_dia(prof, nz) if shaped else spec.stock_dia
            if nz < -0.01 and nz > -spec.part_length - 0.01 and bore_d + 0.5 < nx < mat - 0.01:
                issues.append(Issue(n, "error", f"rapid to X{nx:g} Z{nz:g} is inside material (D{mat:g})"))
            if z > 0 and nz < -0.01 and nx < mat - 0.01 and nx > bore_d + 0.5:
                issues.append(Issue(n, "error", f"rapid Z move into the part at X{nx:g}"))
            moves.append(("rapid", (x, z), (nx, nz)))
            t += math.dist((x / 2, z), (nx / 2, nz)) / RAPID_MM_MIN
        else:
            if feed is None:
                issues.append(Issue(n, "error", "feed move without F"))
            moves.append(("feed", (x, z), (nx, nz)))
            t += feed_time(math.dist((x / 2, z), (nx / 2, nz)), max(x, nx))
        x, z = nx, nz
    tool_changes = len(re.findall(r"^T\d{4}", code, re.M))
    return moves, issues, t + tool_changes * 0.1


def _cut_end(pts, level, internal=False):
    """Leftmost Z the roughing pass at this diameter can reach before hitting the contour."""
    for (xa, za), (xb, zb) in zip(pts, pts[1:]):
        lo, hi = min(xa, xb), max(xa, xb)
        if (not internal and xb > level >= xa - 1e-9 and xb > xa) or (internal and xb < level <= xa + 1e-9 and xb < xa):
            if abs(xb - xa) < 1e-9:
                return za
            return za + (zb - za) * (level - xa) / (xb - xa)
    return pts[-1][1]


# ================================================================= PLOTS
def backplot(moves, spec, path, title):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle, Circle as Circ, Polygon as Poly

    if spec.machine == "lathe":
        fig, ax = plt.subplots(figsize=(11, 4.2))
        L = spec.part_length
        ax.add_patch(Rectangle((-L - 4, 0), L + 5, spec.stock_dia / 2, fc="#e9ecef", ec="#adb5bd", lw=0.8, label="stock"))
        pz = [p[0] for p in spec.profile]
        pr = [p[1] / 2 for p in spec.profile]
        ax.add_patch(Poly(list(zip([0] + pz + [pz[-1]], [0] + pr + [0])), fc="#cfe2ff", ec="#0b5394", lw=1.4, label="finished"))
        if spec.bore:
            ax.add_patch(Rectangle((-spec.bore.depth, 0), spec.bore.depth, spec.bore.dia / 2, fc="white", ec="#0b5394", lw=1.0))
        for kind, a, b in moves:
            (xa, za), (xb, zb) = a, b
            if max(xa, xb) > spec.stock_dia + 40 or max(za, zb) > 60:
                continue
            st = {"rapid": dict(c="#d62728", ls="--", lw=0.6), "feed": dict(c="#2ca02c", lw=0.7),
                  "finish": dict(c="#000000", lw=1.3), "thread": dict(c="#9467bd", lw=1.6)}[kind]
            ax.plot([za, zb], [xa / 2, xb / 2], **st)
        ax.axhline(0, color="#666", lw=0.6, ls="-.")
        ax.set_xlabel("Z (mm)")
        ax.set_ylabel("radius X/2 (mm)")
        ax.set_aspect("equal")
        ax.set_xlim(-L - 10, 12)
        ax.set_ylim(-2, spec.stock_dia / 2 + 8)
    else:
        fig, (ax, ax2) = plt.subplots(1, 2, figsize=(12, 5.5), gridspec_kw={"width_ratios": [3, 2]})
        mx = (spec.stock_x - _w(spec)) / 2
        ys = [p[1] for p in spec.outline] or [0]
        my = (spec.stock_y - (max(ys) - min(ys))) / 2
        ax.add_patch(Rectangle((-mx, -my), spec.stock_x, spec.stock_y, fc="#e9ecef", ec="#adb5bd"))
        if spec.outline:
            ax.add_patch(Poly(spec.outline, fc="#cfe2ff", ec="#0b5394", lw=1.4))
        for h in spec.holes:
            ax.add_patch(Circ((h.x, h.y), h.dia / 2, fc="white", ec="#0b5394"))
        for p in spec.pockets:
            if p.shape == "circle":
                ax.add_patch(Circ((p.cx, p.cy), p.r, fc="#f8f9fa", ec="#0b5394", ls="--"))
            elif p.shape == "rect":
                from matplotlib.patches import FancyBboxPatch
                r = p.corner_r
                ax.add_patch(FancyBboxPatch((p.cx - p.w / 2 + r, p.cy - p.h / 2 + r), p.w - 2 * r, p.h - 2 * r,
                                            boxstyle=f"round,pad={r}", fc="#f8f9fa", ec="#0b5394", ls="--"))
            else:
                ax.add_patch(Poly(p.points, fc="#f8f9fa", ec="#0b5394", ls="--"))
        for kind, a, b in moves:
            if a[2] > 60 or b[2] > 60:
                continue
            st = dict(c="#d62728", ls="--", lw=0.5) if kind == "rapid" else dict(c="#2ca02c", lw=0.6)
            ax.plot([a[0], b[0]], [a[1], b[1]], **st)
            ax2.plot([a[0], b[0]], [a[2], b[2]], **st)
        ax.set_aspect("equal")
        ax.set_title("top view (XY)")
        ax2.set_title("side (XZ)")
        ax2.axhline(0, color="#0b5394", lw=1)
        ax2.axhline(-spec.part_z, color="#0b5394", lw=1)
        ax2.set_ylim(-spec.part_z - 8, 30)
    fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
