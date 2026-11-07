"""DXF reader: exact geometry, the most reliable input. DWG must be converted to
DXF first (free: ODA File Converter, or 'Save As DXF' in any CAD tool)."""
from __future__ import annotations

import math

import ezdxf
from ezdxf import path as ezpath

from .drawing import Drawing, Seg, Circle, Text

CENTER_LT = ("CENTER", "DASHDOT", "CHAIN", "PHANTOM", "ACAD_ISO04", "ACAD_ISO08", "ACAD_ISO10")
HIDDEN_LT = ("HIDDEN", "DASHED", "ACAD_ISO02")
SKIP_LAYERS = ("DIM", "TEXT", "BORDER", "TITLE", "FRAME", "HATCH", "NOTE", "ANNO")


def _kind(e, doc) -> str:
    lt = (e.dxf.get("linetype", "BYLAYER") or "BYLAYER").upper()
    layer = e.dxf.get("layer", "0")
    if lt in ("BYLAYER", "BYBLOCK"):
        try:
            lt = doc.layers.get(layer).dxf.linetype.upper()
        except Exception:
            lt = "CONTINUOUS"
    lname = layer.upper()
    if any(k in lt for k in CENTER_LT) or "CENT" in lname or "AXIS" in lname:
        return "center"
    if any(k in lt for k in HIDDEN_LT) or "HIDDEN" in lname:
        return "hidden"
    if any(k in lname for k in SKIP_LAYERS):
        return "thin"
    return "visible"


def _dim_text(dim) -> str:
    raw = dim.dxf.get("text", "") or ""
    try:
        meas = dim.get_measurement()
        meas = float(meas) if not hasattr(meas, "magnitude") else float(meas.magnitude)
    except Exception:
        meas = None
    dimtype = dim.dimtype & 0x0F
    prefix = ""
    if dimtype == 3:
        prefix = "Ø"
    elif dimtype == 4:
        prefix = "R"
    val = f"{meas:g}" if meas is not None else ""
    if raw in ("", "<>"):
        return prefix + val
    if "<>" in raw:
        return raw.replace("<>", val)
    return raw


def read_dxf(path: str) -> Drawing:
    doc = ezdxf.readfile(path)
    msp = doc.modelspace()
    d = Drawing(source="dxf")
    insunits = doc.header.get("$INSUNITS", 4)
    unit_scale = {1: 25.4, 2: 304.8, 4: 1.0, 5: 10.0, 6: 1000.0}.get(insunits, 1.0)
    if insunits not in (0, 4):
        d.notes.append(f"DXF units code {insunits}, converted to mm (x{unit_scale})")

    for e in msp:
        t = e.dxftype()
        try:
            if t == "LINE":
                s, en = e.dxf.start, e.dxf.end
                d.segs.append(Seg(s.x, s.y, en.x, en.y, _kind(e, doc), e.dxf.layer))
            elif t in ("LWPOLYLINE", "POLYLINE", "SPLINE", "ELLIPSE"):
                k = _kind(e, doc)
                pts = [(p.x, p.y) for p in ezpath.make_path(e).flattening(0.02)]
                for a, b in zip(pts, pts[1:]):
                    if math.dist(a, b) > 1e-6:
                        d.segs.append(Seg(a[0], a[1], b[0], b[1], k, e.dxf.layer))
                closed = getattr(e, "closed", False) or (t == "POLYLINE" and e.is_closed)
                if (closed or (len(pts) > 3 and math.dist(pts[0], pts[-1]) < 1e-4)) and k == "visible":
                    d.polys.append([list(p) for p in pts])
            elif t == "CIRCLE":
                c = e.dxf.center
                d.circles.append(Circle(c.x, c.y, e.dxf.radius, _kind(e, doc)))
            elif t == "ARC":
                sweep = (e.dxf.end_angle - e.dxf.start_angle) % 360
                k = _kind(e, doc)
                if sweep >= 250:              # 3/4 arc = thread major dia convention
                    c = e.dxf.center
                    d.circles.append(Circle(c.x, c.y, e.dxf.radius, "thin" if k == "visible" else k, partial=True))
                else:
                    pts = [(p.x, p.y) for p in ezpath.make_path(e).flattening(0.02)]
                    for a, b in zip(pts, pts[1:]):
                        d.segs.append(Seg(a[0], a[1], b[0], b[1], k, e.dxf.layer))
            elif t in ("TEXT", "ATTRIB"):
                p = e.dxf.insert
                d.texts.append(Text(e.dxf.text, p.x, p.y, e.dxf.get("height", 3.5)))
            elif t == "MTEXT":
                p = e.dxf.insert
                d.texts.append(Text(e.plain_text().replace("\n", " "), p.x, p.y, e.dxf.get("char_height", 3.5)))
            elif t == "DIMENSION":
                p = e.dxf.get("text_midpoint", None) or e.dxf.defpoint
                d.texts.append(Text(_dim_text(e), p.x, p.y))
        except Exception as ex:                     # never die on one odd entity
            d.notes.append(f"skipped {t}: {ex}")

    if unit_scale != 1.0:
        d.transform(unit_scale)
    return d
