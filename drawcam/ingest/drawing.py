"""Common in-memory drawing model produced by every reader (DXF, PDF, scan)."""
from __future__ import annotations

import math
from dataclasses import dataclass, field


@dataclass
class Seg:
    x1: float
    y1: float
    x2: float
    y2: float
    kind: str = "visible"           # visible | center | hidden | thin
    layer: str = ""

    @property
    def length(self) -> float:
        return math.hypot(self.x2 - self.x1, self.y2 - self.y1)

    def is_horizontal(self, tol=1e-3) -> bool:
        return abs(self.y2 - self.y1) <= tol * max(1.0, self.length)

    def is_vertical(self, tol=1e-3) -> bool:
        return abs(self.x2 - self.x1) <= tol * max(1.0, self.length)


@dataclass
class Circle:
    cx: float
    cy: float
    r: float
    kind: str = "visible"
    partial: bool = False           # e.g. 3/4 thread arc


@dataclass
class Text:
    text: str
    x: float
    y: float
    h: float = 3.5
    conf: float = 1.0               # OCR confidence 0..1 (1 for CAD text)


@dataclass
class Drawing:
    source: str                     # dxf | pdf | scan
    segs: list = field(default_factory=list)
    circles: list = field(default_factory=list)
    polys: list = field(default_factory=list)      # closed polylines [[x,y],...]
    texts: list = field(default_factory=list)
    scale_known: bool = True        # False when geometry units had to be guessed
    notes: list = field(default_factory=list)      # reader warnings

    def visible(self):
        return [s for s in self.segs if s.kind == "visible"]

    def bbox(self, kinds=("visible",)):
        xs, ys = [], []
        for s in self.segs:
            if s.kind in kinds:
                xs += [s.x1, s.x2]
                ys += [s.y1, s.y2]
        for c in self.circles:
            if c.kind in kinds:
                xs += [c.cx - c.r, c.cx + c.r]
                ys += [c.cy - c.r, c.cy + c.r]
        if not xs:
            return (0, 0, 0, 0)
        return (min(xs), min(ys), max(xs), max(ys))

    def transform(self, sx: float, dx: float = 0.0, dy: float = 0.0):
        """Scale every coordinate by sx then shift (used to convert paper/pixel units to mm)."""
        for s in self.segs:
            s.x1, s.y1, s.x2, s.y2 = s.x1 * sx + dx, s.y1 * sx + dy, s.x2 * sx + dx, s.y2 * sx + dy
        for c in self.circles:
            c.cx, c.cy, c.r = c.cx * sx + dx, c.cy * sx + dy, c.r * sx
        self.polys = [[[x * sx + dx, y * sx + dy] for x, y in p] for p in self.polys]
        for t in self.texts:
            t.x, t.y, t.h = t.x * sx + dx, t.y * sx + dy, t.h * sx

    def all_text(self) -> str:
        return "\n".join(t.text for t in self.texts)
