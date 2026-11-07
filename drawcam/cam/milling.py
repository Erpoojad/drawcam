"""MillSpec -> Fanuc 0i-M style VMC program.

Operation order: face, pockets (offset rings, ramp/helix entry), outer contour,
spot drill, drill (G81/G83), tap (rigid G84), ream (G85).
X0 Y0 = bottom-left corner of the finished part, Z0 = finished top face.
"""
from __future__ import annotations

import math

from shapely.geometry import Polygon, Point, box
from shapely.geometry.polygon import orient

from ..spec import MillSpec, ReviewItem
from .gcode import GCode, f
from .shop import rpm, tap_drill

RAMP_DEG = 2.0


class Mill:
    def __init__(self, spec: MillSpec, cfg):
        self.s = spec
        self.cfg = cfg
        self.M = cfg["mill"]
        self.m = cfg["materials"][spec.material]
        self.g = GCode()
        self.safe = self.M["safe_z"]
        self.max_rpm = self.M["max_rpm"]
        self.soft = spec.material in ("AL", "BRASS")
        xs = [p[0] for p in spec.outline] or [0, spec.stock_x - 4]
        ys = [p[1] for p in spec.outline] or [0, spec.stock_y - 4]
        self.pw, self.ph = max(xs) - min(xs), max(ys) - min(ys)
        self.mx = (spec.stock_x - self.pw) / 2
        self.my = (spec.stock_y - self.ph) / 2
        self.ztop = spec.stock_z - spec.part_z          # stock above Z0

    # ------------------------------------------------------------ helpers
    def add(self, field, msg, sev="warn"):
        if not any(r.field == field and r.message == msg for r in self.s.review):
            self.s.review.append(ReviewItem(field, msg, sev))

    def tool_change(self, t, desc, s, x, y):
        g = self.g
        g.blank()
        g.comment(f"T{t} {desc}")
        g(f"T{t:02d} M06")
        g("G00 G90", self.M["work_offset"], f"X{f(x)}", f"Y{f(y)}", f"S{s}", "M03")
        g("G43", f"H{t:02d}", f"Z{f(self.safe)}", "M08")

    def end_tool(self):
        self.g(f"G00 Z{f(self.safe)}")
        self.g("M09")
        self.g("M05")
        self.g("G91 G28 Z0.")
        self.g("G90")
        self.g("M01")

    def endmill_for(self, max_dia):
        sizes = sorted((float(k), v) for k, v in self.M["endmills"].items())
        ok = [(d, t) for d, t in sizes if d <= max_dia + 1e-6]
        return ok[-1] if ok else None

    def feed_em(self, d, s):
        return round(s * self.m["fz"] * self.M["endmill_flutes"] * (d / 10) ** 0.3)

    # --------------------------------------------------------------- face
    def face(self):
        if self.ztop <= 0.01:
            return
        t = self.M["tools"]["face"]
        D, R = t["dia"], t["dia"] / 2
        s = rpm(self.m["vc_face"], D, self.max_rpm)
        fz = round(s * self.m["fz_face"] * t["flutes"])
        x0, x1 = -self.mx - R - 5, self.pw + self.mx + R + 5
        y0, y1 = -self.my, self.ph + self.my
        so = 0.7 * D
        lanes = []
        y = y0 + R - so * 0.3
        while True:
            lanes.append(y)
            if y + R >= y1:
                break
            y += so
        self.tool_change(t["t"], t["desc"], s, x0, lanes[0])
        levels = []
        z = self.ztop
        while z > 0.3 + 1e-6:
            z = max(0.3, z - 1.5)
            levels.append(z)
        levels.append(0.0)
        g = self.g
        for z in levels:
            g("G00", f"X{f(x0)}", f"Y{f(lanes[0])}")
            g("G00", f"Z{f(self.ztop + 2)}")
            g("G01", f"Z{f(z)}", f"F{fz}")
            fwd = True
            for i, y in enumerate(lanes):
                if i:
                    g("G01", f"Y{f(y)}")
                g("G01", f"X{f(x1 if fwd else x0)}")
                fwd = not fwd
            g("G00", f"Z{f(self.ztop + 5)}")
        self.end_tool()

    # ------------------------------------------------------------ pockets
    def pocket_poly(self, p):
        if p.shape == "rect":
            r = p.corner_r
            b = box(p.cx - p.w / 2 + r, p.cy - p.h / 2 + r, p.cx + p.w / 2 - r, p.cy + p.h / 2 - r)
            return b.buffer(r, quad_segs=16) if r > 0 else box(p.cx - p.w / 2, p.cy - p.h / 2, p.cx + p.w / 2, p.cy + p.h / 2)
        if p.shape == "circle":
            return Point(p.cx, p.cy).buffer(p.r, quad_segs=32)
        return Polygon(p.points)

    def pocket_tool(self, p, poly):
        minx, miny, maxx, maxy = poly.bounds
        width = min(maxx - minx, maxy - miny)
        lim = 0.6 * width if p.shape != "circle" else p.r          # <= half the pocket diameter
        if p.shape == "rect" and p.corner_r > 0:
            lim = min(lim, 1.6 * p.corner_r)       # tool radius <= 0.8 x corner radius: no full wrap in corners
        if p.shape == "poly":
            lim = min(lim, 10)
        return self.endmill_for(lim)

    def pockets(self):
        for p in self.s.pockets:
            poly = self.pocket_poly(p)
            sel = self.pocket_tool(p, poly)
            if not sel:
                self.add("pocket", f"no end mill small enough for pocket at X{p.cx:g} Y{p.cy:g}", "block")
                continue
            D, t = sel
            if p.shape == "circle":
                self.circle_pocket(p, D, t)
            else:
                self.ring_pocket(p, poly, D, t)

    def _levels(self, depth, D):
        doc = (1.0 if self.soft else 0.5) * D
        n = max(1, math.ceil(depth / doc - 1e-9))
        return [-(depth * (i + 1) / n) for i in range(n)]

    def circle_pocket(self, p, D, t):
        """Helical entry, then concentric G03 circles out to size (climb, CCW)."""
        R = D / 2
        s = rpm(self.m["vc_mill"], D, self.max_rpm)
        fe = self.feed_em(D, s)
        rmax = p.r - R
        if rmax <= 0:
            self.add("pocket", f"circle Ø{2 * p.r:g} too small for D{D:g}", "block")
            return
        so = (0.6 if self.soft else 0.4) * D
        radii = []
        r = min(rmax, max(0.3 * D, 0.5))
        while r < rmax - 1e-6:
            radii.append(r)
            r += so
        radii.append(rmax)
        helix_r = radii[0]
        z_top = 0.0
        self.tool_change(t, f"END MILL D{D:g}  CIRCULAR POCKET D{2 * p.r:g}", s, p.cx + helix_r, p.cy)
        g = self.g
        g.comment(f"POCKET D{2 * p.r:g} DEPTH {p.depth:g}")
        g("G00", f"Z{f(z_top + 2)}")
        pitch = 2 * math.pi * helix_r * math.tan(math.radians(RAMP_DEG))
        z = z_top
        g("G01", f"Z{f(z_top)}", f"F{fe // 2}")
        for zl in self._levels(p.depth, D):
            while z > zl + 1e-6:
                z = max(zl, z - pitch)
                g("G03", f"X{f(p.cx + helix_r)}", f"Y{f(p.cy)}", f"Z{f(z)}", f"I{f(-helix_r)}", "J0.", f"F{fe // 2}")
            g("G03", f"X{f(p.cx + helix_r)}", f"Y{f(p.cy)}", f"I{f(-helix_r)}", "J0.", f"F{fe}")   # clean the floor
            for r in radii[1:]:
                g("G01", f"X{f(p.cx + r)}", f"Y{f(p.cy)}")
                g("G03", f"X{f(p.cx + r)}", f"Y{f(p.cy)}", f"I{f(-r)}", "J0.")
            g("G01", f"X{f(p.cx + helix_r)}", f"Y{f(p.cy)}")
        self.end_tool()

    def ring_pocket(self, p, poly, D, t):
        R = D / 2
        s = rpm(self.m["vc_mill"], D, self.max_rpm)
        fe = self.feed_em(D, s)
        so = (0.6 if self.soft else 0.45) * D
        rings = []
        off = R
        while True:
            inner = poly.buffer(-off, quad_segs=8)
            if inner.is_empty:
                break
            geoms = list(inner.geoms) if inner.geom_type == "MultiPolygon" else [inner]
            if len(geoms) > 1:
                self.add("pocket", "pocket splits into islands at tool offset; check the inner path", "warn")
            ring = orient(geoms[0].simplify(0.005), 1.0).exterior.coords[:]
            rings.append(ring)
            off += so
        if not rings:
            self.add("pocket", f"pocket at X{p.cx:g} Y{p.cy:g} narrower than D{D:g}", "block")
            return
        rings = rings[::-1]                         # innermost first, finish at the wall
        start = rings[0][0]
        self.tool_change(t, f"END MILL D{D:g}  POCKET", s, start[0], start[1])
        g = self.g
        label = f"{p.w:g}X{p.h:g} R{p.corner_r:g}" if p.shape == "rect" else "PROFILE"
        g.comment(f"POCKET {label} DEPTH {p.depth:g}")
        g("G00", "Z2.")
        z = 0.0
        g("G01", "Z0.", f"F{fe // 2}")
        a, b = rings[0][0], rings[0][1]
        L = math.dist(a, b)
        for zl in self._levels(p.depth, D):
            # ramp back and forth along the first edge of the inner ring
            if L >= 0.5 * D:
                dz = L * math.tan(math.radians(RAMP_DEG))
                at_a = True
                while z > zl + 1e-6:
                    z = max(zl, z - dz)
                    tgt = b if at_a else a
                    g("G01", f"X{f(tgt[0])}", f"Y{f(tgt[1])}", f"Z{f(z)}", f"F{fe // 2}")
                    at_a = not at_a
                if not at_a:
                    g("G01", f"X{f(a[0])}", f"Y{f(a[1])}")
            else:
                self.add("pocket", "pocket too small to ramp: plunging; needs centre-cutting end mill")
                z = zl
                g("G01", f"Z{f(z)}", f"F{fe // 4}")
            g(f"F{fe}")
            for ring in rings:
                # start each ring at the vertex nearest to where we are
                i = min(range(len(ring) - 1), key=lambda k: math.dist(ring[k], a))
                seq = ring[i:-1] + ring[:i] + [ring[i]]
                for x, y in seq:
                    g("G01", f"X{f(x)}", f"Y{f(y)}")
                a = seq[-1]
            g("G01", f"X{f(rings[0][0][0])}", f"Y{f(rings[0][0][1])}")
            a = rings[0][0]
        self.end_tool()

    # ------------------------------------------------------------ contour
    def contour(self):
        sel = self.endmill_for(12)
        D, t = sel
        R = D / 2
        part = Polygon(self.s.outline)
        path = orient(part.buffer(R, quad_segs=8), -1.0).exterior.coords[:]   # CW = climb outside
        s = rpm(self.m["vc_mill"], D, self.max_rpm)
        fe = self.feed_em(D, s)
        depth = self.s.part_z + 0.3
        self.add("contour", f"outer contour is cut to Z-{depth:g}: part must sit on parallels clear of the vise, "
                            "or delete this section and finish the sides in a second setup")
        i = min(range(len(path) - 1), key=lambda k: path[k][0] + path[k][1])
        seq = path[i:-1] + path[:i] + [path[i]]
        sx, sy = seq[0]
        lead = (sx - self.mx - D - 3, sy)
        self.tool_change(t, f"END MILL D{D:g}  OUTER CONTOUR", s, *lead)
        g = self.g
        g.comment("OUTER CONTOUR - CLIMB, COMPUTED PATH [NO G41]")
        doc = min(1.5 * D, 15.0) if self.soft else min(1.0 * D, 10.0)
        n = max(1, math.ceil((depth + self.ztop) / doc))
        for k in range(1, n + 1):
            z = self.ztop - (depth + self.ztop) * k / n
            g("G00", f"X{f(lead[0])}", f"Y{f(lead[1])}")
            g("G00", f"Z{f(self.ztop + 2)}")
            g("G01", f"Z{f(z)}", f"F{fe // 2}")
            g("G01", f"X{f(sx)}", f"Y{f(sy)}", f"F{fe}")
            for x, y in seq[1:]:
                g("G01", f"X{f(x)}", f"Y{f(y)}")
            g("G01", f"X{f(lead[0])}", f"Y{f(sy)}")
            g("G00", f"Z{f(self.ztop + 2)}")
        self.end_tool()

    # -------------------------------------------------------------- holes
    def hole_top(self, h):
        for p in self.s.pockets:
            if self.pocket_poly(p).buffer(0.01).contains(Point(h.x, h.y)):
                return -p.depth
        return 0.0

    def cycle_group(self, holes, t, desc, s, cycle_line):
        """cycle_line(h, ztop) -> list of words for the first hole; following holes get XY only."""
        g = self.g
        self.tool_change(t, desc, s, holes[0].x, holes[0].y)
        g("G00", "Z10.")
        last = None
        for h in holes:
            zt = self.hole_top(h)
            words = cycle_line(h, zt)
            key = tuple(words)
            if last is None or key != last:
                g("G98", *words[:1], f"X{f(h.x)}", f"Y{f(h.y)}", *words[1:])
                last = key
            else:
                g(f"X{f(h.x)}", f"Y{f(h.y)}")
        g("G80")
        self.end_tool()

    def holes(self):
        holes = self.s.holes
        if not holes:
            return
        m = self.m
        # spot every hole
        sp = self.M["tools"]["spot"]
        s = rpm(m["vc_drill"] * 2.5, sp["dia"] / 2, self.max_rpm)
        self.cycle_group(sorted(holes, key=lambda h: (h.y, h.x)), sp["t"], sp["desc"], s,
                         lambda h, zt: ["G81", f"Z{f(zt - min(h.dia, sp['dia']) * 0.3)}", f"R{f(zt + 2)}", f"F{round(s * 0.05)}"])
        # drill sizes per hole
        plan = []
        for h in holes:
            if h.kind == "tap":
                d = tap_drill(h.dia, h.pitch)
            elif h.kind == "ream":
                d = round(h.dia - (0.2 if h.dia <= 10 else 0.3), 1)
            else:
                d = round(h.dia, 2)
                if abs(d * 2 - round(d * 2)) > 0.01:
                    self.add("drill", f"Ø{d} is not a standard drill size; drill under and ream/bore, or confirm the drill")
            plan.append((d, h))
        t = self.M["first_drill_t"]
        for d in sorted({d for d, _ in plan}):
            grp = [h for dd, h in plan if dd == d]
            s = rpm(m["vc_drill"], d, self.max_rpm)
            feed = round(s * m["f_drill"] * min(1.0, d / 10) ** 0.5)

            def line(h, zt, d=d, feed=feed):
                tip = 0.3 * d
                if h.through:
                    z = zt - (self.s.part_z + zt) - tip - 1.0       # break through bottom face
                elif h.kind == "tap":
                    full = h.depth + 3 * h.pitch                       # usable thread + tap lead
                    floor = (self.s.part_z + zt) - 1.0 - tip           # keep 1 mm of floor under the drill point
                    if full > floor:
                        full = floor
                        if full < h.depth + 1.5 * h.pitch:
                            self.add("tap", f"M{h.dia:g} DP{h.depth:g}: not enough material below for a plug tap; use a bottoming tap")
                    z = zt - full - tip
                else:
                    z = zt - h.depth - tip
                cyc = ["G83", f"Z{f(z)}", f"R{f(zt + 2)}", f"Q{f(d)}", f"F{feed}"] if (zt - z) > 3 * d \
                    else ["G81", f"Z{f(z)}", f"R{f(zt + 2)}", f"F{feed}"]
                return cyc
            self.cycle_group(sorted(grp, key=lambda h: (h.y, h.x)), t, f"DRILL D{d:g} HSS", s, line)
            t += 1
        if any(h.through for h in holes):
            self.add("holes", "through holes break out under the part: use parallels with clearance below every hole")
        # taps
        t = self.M["first_tap_t"]
        for key in sorted({(h.dia, h.pitch) for h in holes if h.kind == "tap"}):
            grp = [h for h in holes if h.kind == "tap" and (h.dia, h.pitch) == key]
            s = rpm(m["vc_tap"], key[0], min(self.max_rpm, 800))

            def tline(h, zt, s=s, key=key):
                z = zt - (self.s.part_z + zt) - 2 * key[1] if h.through else zt - h.depth
                return ["G84", f"Z{f(z)}", f"R{f(zt + 5)}", f"F{f(s * key[1], 2)}"]
            self.g.blank()
            self.g.comment("RIGID TAP: M29 BEFORE G84. FLOATING HOLDER: DELETE M29")
            first = len(self.g.lines)
            self.cycle_group(sorted(grp, key=lambda h: (h.y, h.x)), t, f"TAP M{key[0]:g}X{key[1]:g}", s, tline)
            # insert M29 right before the first G84 block
            for i in range(first, len(self.g.lines)):
                if self.g.lines[i].startswith("G98 G84"):
                    self.g.lines.insert(i, f"M29 S{s}")
                    break
            t += 1
        # reams
        t = self.M["first_ream_t"]
        for d in sorted({h.dia for h in holes if h.kind == "ream"}):
            grp = [h for h in holes if h.kind == "ream" and h.dia == d]
            s = rpm(m["vc_tap"] * 1.2, d, self.max_rpm)

            def rline(h, zt, d=d, s=s):
                z = zt - (self.s.part_z + zt) - 1.0 if h.through else zt - h.depth
                return ["G85", f"Z{f(z)}", f"R{f(zt + 2)}", f"F{round(s * 0.2)}"]
            self.cycle_group(sorted(grp, key=lambda h: (h.y, h.x)), t, f"REAMER D{d:g} H7", s, rline)
            t += 1

    # ------------------------------------------------------------- driver
    def plan_checks(self):
        tr = self.M["travel"]
        if self.s.stock_x > tr[0] or self.s.stock_y > tr[1]:
            self.add("stock", "stock larger than machine travel", "block")
        for p in self.s.pockets:
            if p.depth > 0.9 * self.s.part_z:
                self.add("pocket", f"pocket depth {p.depth:g} close to/through the part thickness; check fixturing")
        for h in self.s.holes:
            if h.kind == "tap" and not h.through and h.depth < 1.5 * h.dia:
                self.add("tap", f"M{h.dia:g} blind tap depth {h.depth:g} is shallow; confirm thread depth")
        if self.s.outline:
            self.add("contour", f"outer contour is cut to Z-{self.s.part_z + 0.3:g}: part must sit on parallels clear of the vise, "
                                "or delete this section and finish the sides in a second setup")

    def program(self) -> str:
        self.plan_checks()
        # dry run the body first so every warning lands in the header
        body = Mill(self.s, self.cfg)
        body.face(); body.pockets()
        if self.s.outline:
            body.contour()
        body.holes()
        g = self.g
        s = self.s
        g("%")
        g(f"O{self.M['program_number']:04d}")
        g.comment(f"{s.part_name} - VMC")
        g.comment(f"MATERIAL {s.material} STOCK {s.stock_x:g}X{s.stock_y:g}X{s.stock_z:g}")
        g.comment(f"X0Y0 = PART BOTTOM-LEFT CORNER [STOCK EDGE AT X-{self.mx:g} Y-{self.my:g}], Z0 = FINISHED TOP")
        g.comment("GENERATED BY DRAWCAM - PROVE OUT IN SINGLE BLOCK WITH OVERRIDES LOW")
        for r in s.review:
            g.comment(f"CHECK: {r.message}")
        g("G21 G17 G40 G49 G80 G90")
        g("G91 G28 Z0.")
        g("G90")
        g.lines += body.g.lines
        g("G91 G28 Y0.")
        g("G90")
        g("M30")
        g("%")
        return g.text()


def generate_milling(spec: MillSpec, cfg) -> dict:
    return {"OP1": Mill(spec, cfg).program()}
