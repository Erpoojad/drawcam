"""End-to-end and safety tests. Run: python -m pytest -q"""
import os
import re

import pytest

from drawcam.cam.milling import generate_milling
from drawcam.cam.shop import load_shop, tap_drill
from drawcam.cam.turning import generate_turning
from drawcam.ingest.dim_parser import fit_limits, parse
from drawcam.ingest.load import load_drawing
from drawcam.recognize.pipeline import recognize
from drawcam.spec import blocking
from drawcam.verify.sim import sim_lathe, sim_mill

S = os.path.join(os.path.dirname(__file__), "..", "samples")
CFG = load_shop()
FORMATS = [".dxf", ".pdf", "_scan.png"]

SHAFT = [[0.0, 19.85], [-25.0, 19.85], [-25.0, 25.0], [-57.0, 25.0], [-57.0, 40.0], [-72.0, 40.0], [-72.0, 30.0], [-100.0, 30.0]]
BUSH = [[0.0, 34.985], [-30.0, 34.985], [-30.0, 60.0], [-40.0, 60.0]]


@pytest.fixture(scope="module")
def specs():
    out = {}
    for name in ("shaft", "bush", "plate"):
        for ext in FORMATS:
            out[name + ext] = recognize(load_drawing(os.path.join(S, name + ext)))
    return out


# ------------------------------------------------------------- parsing
@pytest.mark.parametrize("text,check", [
    ("Ø25 ±0.01", lambda c: c.dia == 25 and c.tol == (0.01, -0.01)),
    ("M20x1.5", lambda c: c.thread == (20, 1.5)),
    ("2X M8 DP 15", lambda c: c.thread == (8, 1.25) and c.depth == 15 and c.count == 2),
    ("4X Ø8.5 THRU", lambda c: c.dia == 8.5 and c.through and c.count == 4),
    ("Ø20 H7 THRU", lambda c: c.fit == "H7" and abs(c.mid_tol_value(20) - 20.0105) < 1e-6),
    ("%%c35 +0.0/-0.03", lambda c: c.dia == 35 and abs(c.mid_tol_value(35) - 34.985) < 1e-9),
    ("C1", lambda c: c.chamfer == 1),
])
def test_parse(text, check):
    assert check(parse(text))


def test_ocr_fixes():
    assert parse("@25", ocr=True).dia == 25
    assert parse("O40", ocr=True).dia == 40
    assert parse("Cl", ocr=True).chamfer == 1


def test_fits_and_tap_drills():
    assert fit_limits(25, "h6") == (0.0, -0.013)
    assert fit_limits(20, "H7") == (0.021, 0.0)
    assert tap_drill(8, 1.25) == 6.8 and tap_drill(12, 1.75) == 10.2


# ------------------------------------------------------- recognition
@pytest.mark.parametrize("ext", FORMATS)
def test_shaft(specs, ext):
    s = specs["shaft" + ext]
    assert s.machine == "lathe" and s.material == "EN8"
    assert s.profile == SHAFT
    assert len(s.threads) == 1 and s.threads[0].pitch == 1.5 and s.threads[0].z_end == -22
    assert len(s.grooves) == 1 and s.grooves[0].bottom_dia == 16.9
    assert not blocking(s)


@pytest.mark.parametrize("ext", FORMATS)
def test_bush(specs, ext):
    s = specs["bush" + ext]
    assert s.machine == "lathe" and s.profile == BUSH
    assert s.bore and s.bore.through and abs(s.bore.dia - 20.0105) < 1e-6


@pytest.mark.parametrize("ext", FORMATS)
def test_plate(specs, ext):
    s = specs["plate" + ext]
    assert s.machine == "mill" and s.material == "AL" and s.part_z == 20
    kinds = sorted((h.kind, h.dia, h.x, h.y) for h in s.holes)
    assert kinds == sorted([("tap", 8, 30, 70), ("tap", 8, 90, 70), ("ream", 10.0, 100, 40)] +
                           [("drill", 8.5, x, y) for x, y in ((10, 10), (110, 10), (10, 70), (110, 70))])
    rect = [p for p in s.pockets if p.shape == "rect"][0]
    assert (rect.w, rect.h, rect.corner_r, rect.depth) == (40, 30, 5, 8)
    assert [p.r for p in s.pockets if p.shape == "circle"] == [15]


# --------------------------------------------------------- G-code
@pytest.mark.parametrize("name", ["shaft", "bush", "plate"])
def test_gcode_passes_checks(specs, name):
    s = specs[name + ".dxf"]
    progs = generate_turning(s, CFG) if s.machine == "lathe" else generate_milling(s, CFG)
    for op, code in progs.items():
        if s.machine == "lathe":
            _, issues, minutes = sim_lathe(code, s, CFG, flipped=(op == "OP2"))
        else:
            _, issues, minutes = sim_mill(code, s, CFG)
        assert not [i for i in issues if i.severity == "error"], issues
        assert 0.3 < minutes < 30
        assert code.startswith("%\nO") and code.rstrip().endswith("%")
        assert "M30" in code
        # Fanuc: every coordinate must carry a decimal point
        for w in re.findall(r"\b[XZYIJR](-?\d+)(?![\d.])", re.sub(r"\(.*?\)", "", code)):
            pytest.fail(f"integer coordinate {w}")


def test_checker_catches_rapid_into_part(specs):
    s = specs["bush.dxf"]
    code = generate_turning(s, CFG)["OP1"]
    bad = code.replace("G00 X67. Z-43. M08", "G00 X40. Z-35. M08")      # rapid into the Ø60 flange
    _, issues, _ = sim_lathe(bad, s, CFG)
    assert any(i.severity == "error" for i in issues)


def test_checker_catches_mill_rapid_plunge(specs):
    s = specs["plate.dxf"]
    code = generate_milling(s, CFG)["OP1"]
    bad = code.replace("G00 Z2.\nG01 Z0.", "G00 Z-3.\nG01 Z0.", 1)
    _, issues, _ = sim_mill(bad, s, CFG)
    assert any(i.severity == "error" for i in issues)


def test_threads_feed_and_relief(specs):
    code = generate_turning(specs["shaft.dxf"], CFG)["OP1"]
    assert "G76 X18.16 Z-23.5 P920" in code          # minor dia, ends mid-relief, 0.92 mm depth
    assert "F1.5" in code


def test_blocked_spec_refuses(tmp_path, specs):
    from drawcam.cli import main
    from drawcam.spec import ReviewItem, to_json
    s = specs["plate.dxf"]
    s.review.append(ReviewItem("material", "test block", "block"))
    p = tmp_path / "spec.json"
    to_json(s, str(p))
    assert main(["gcode", str(p), "-o", str(tmp_path), "--no-log"]) == 1
    assert not list(tmp_path.glob("*.nc"))
    s.review.pop()
