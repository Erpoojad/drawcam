"""Reviewable part specification.

The spec is the contract between "reading the drawing" (uncertain, ML-assisted)
and "writing G-code" (deterministic). Every value the reader was not sure about
is listed in `review`. G-code is only generated from a spec, never straight from
an image, so a human can check and correct the JSON first.

Units: millimetres. Lathe: Z0 = finished right face, Z negative into the part,
X values are DIAMETERS. Mill: X0 Y0 = bottom-left corner of stock, Z0 = top face.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from typing import Optional


@dataclass
class ReviewItem:
    field: str
    message: str
    severity: str = "warn"          # "warn" = check it, "block" = cannot cut until fixed


# ----------------------------------------------------------------- lathe ----
@dataclass
class Thread:
    z_start: float                  # nearer the face (less negative)
    z_end: float
    major_dia: float
    pitch: float
    external: bool = True


@dataclass
class Groove:
    z: float                        # left edge of groove (more negative side)
    width: float
    bottom_dia: float


@dataclass
class Bore:
    dia: float
    depth: float                    # positive number, measured from Z0
    through: bool = False


@dataclass
class TurnSpec:
    part_name: str
    material: str
    part_length: float
    max_dia: float
    stock_dia: float
    stock_length: float
    # Finished outer profile, from Z0 leftwards: list of [z, dia]
    profile: list = field(default_factory=list)
    threads: list = field(default_factory=list)
    grooves: list = field(default_factory=list)
    bore: Optional[Bore] = None
    chamfer: float = 0.5            # default edge break
    review: list = field(default_factory=list)
    machine: str = "lathe"


# ------------------------------------------------------------------ mill ----
@dataclass
class Hole:
    x: float
    y: float
    dia: float
    depth: float                    # positive; for through holes = stock thickness
    kind: str = "drill"             # drill | tap | ream | counterbore
    pitch: float = 0.0              # taps only
    through: bool = False


@dataclass
class Pocket:
    shape: str                      # rect | circle | poly
    depth: float
    cx: float = 0.0
    cy: float = 0.0
    w: float = 0.0
    h: float = 0.0
    r: float = 0.0                  # circle radius
    corner_r: float = 0.0
    points: list = field(default_factory=list)   # poly


@dataclass
class MillSpec:
    part_name: str
    material: str
    stock_x: float
    stock_y: float
    stock_z: float
    part_z: float                   # finished thickness
    outline: list = field(default_factory=list)  # closed [x, y] list, part profile
    holes: list = field(default_factory=list)
    pockets: list = field(default_factory=list)
    review: list = field(default_factory=list)
    machine: str = "mill"


# ------------------------------------------------------------------- I/O ----
def to_json(spec, path: str) -> None:
    with open(path, "w") as f:
        json.dump(asdict(spec), f, indent=2)


def from_json(path: str):
    with open(path) as f:
        d = json.load(f)
    d["review"] = [ReviewItem(**r) for r in d.get("review", [])]
    if d.get("machine") == "lathe":
        d["threads"] = [Thread(**t) for t in d.get("threads", [])]
        d["grooves"] = [Groove(**g) for g in d.get("grooves", [])]
        d["bore"] = Bore(**d["bore"]) if d.get("bore") else None
        return TurnSpec(**d)
    d["holes"] = [Hole(**h) for h in d.get("holes", [])]
    d["pockets"] = [Pocket(**p) for p in d.get("pockets", [])]
    return MillSpec(**d)


def blocking(spec) -> list:
    return [r for r in spec.review if r.severity == "block"]
