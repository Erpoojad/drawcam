"""Shop setup: machines, turret/magazine, cutting data.

THESE NUMBERS ARE STARTING POINTS, NOT YOUR SHOP'S NUMBERS. Copy
`shop_config.example.json`, put in your real turret positions, offsets,
max RPM and the speeds your operators actually use, and pass it with --shop.
"""
from __future__ import annotations

import copy
import json
import math

DEFAULT = {
    "lathe": {
        "controller": "fanuc-0i-T",
        "max_rpm": 3000,
        "max_dia": 250,
        "program_number": 1001,
        "tools": {
            "face_rough": {"t": 1, "desc": "CNMG 120408 80deg rough/face", "nose_r": 0.8},
            "finish":     {"t": 3, "desc": "VNMG 160404 35deg finish", "nose_r": 0.4},
            "groove":     {"t": 5, "desc": "Grooving insert 3.0 mm, left edge reference", "width": 3.0},
            "thread":     {"t": 7, "desc": "External thread insert 60deg"},
            "drill":      {"t": 9, "desc": "Drill on axis (set size per job)"},
            "bore":       {"t": 10, "desc": "Boring bar min bore 16", "min_bore": 16.0, "nose_r": 0.4},
            "part":       {"t": 11, "desc": "Parting blade 3.0 mm, left edge reference", "width": 3.0},
        },
    },
    "mill": {
        "controller": "fanuc-0i-M",
        "max_rpm": 8000,
        "travel": [600, 400, 450],
        "program_number": 2001,
        "work_offset": "G54",
        "tools": {
            "face":      {"t": 1, "dia": 50.0, "flutes": 4, "desc": "Face mill D50 carbide"},
            "spot":      {"t": 2, "dia": 10.0, "desc": "Spot drill D10 90deg"},
        },
        # end mill diameter -> pocket number (4-flute carbide, centre cutting)
        "endmills": {"4": 4, "6": 6, "8": 8, "10": 10, "12": 12, "16": 16, "20": 20},
        "endmill_flutes": 4,
        "first_drill_t": 21,       # drills, taps, reamers numbered per job from here
        "first_tap_t": 31,
        "first_ream_t": 41,
        "safe_z": 25.0,
    },
    # Vc in m/min, feeds: turning mm/rev, milling fz mm/tooth, drill mm/rev.
    # Carbide turning/milling, HSS drills and taps (typical Indian job-shop kit).
    "materials": {
        "MS":    {"vc_rough": 180, "vc_finish": 220, "f_rough": 0.25, "f_finish": 0.10, "doc": 2.0,
                  "vc_mill": 120, "fz": 0.05, "vc_face": 150, "fz_face": 0.12, "vc_drill": 25, "f_drill": 0.12, "vc_tap": 8, "vc_thread": 100},
        "EN8":   {"vc_rough": 160, "vc_finish": 200, "f_rough": 0.22, "f_finish": 0.10, "doc": 1.8,
                  "vc_mill": 100, "fz": 0.045, "vc_face": 130, "fz_face": 0.10, "vc_drill": 20, "f_drill": 0.10, "vc_tap": 6, "vc_thread": 90},
        "EN24":  {"vc_rough": 120, "vc_finish": 150, "f_rough": 0.20, "f_finish": 0.08, "doc": 1.5,
                  "vc_mill": 80, "fz": 0.04, "vc_face": 100, "fz_face": 0.08, "vc_drill": 14, "f_drill": 0.08, "vc_tap": 4, "vc_thread": 70},
        "SS304": {"vc_rough": 140, "vc_finish": 170, "f_rough": 0.20, "f_finish": 0.08, "doc": 1.5,
                  "vc_mill": 80, "fz": 0.035, "vc_face": 100, "fz_face": 0.08, "vc_drill": 12, "f_drill": 0.08, "vc_tap": 4, "vc_thread": 70},
        "AL":    {"vc_rough": 300, "vc_finish": 400, "f_rough": 0.30, "f_finish": 0.12, "doc": 3.0,
                  "vc_mill": 250, "fz": 0.06, "vc_face": 400, "fz_face": 0.15, "vc_drill": 60, "f_drill": 0.15, "vc_tap": 15, "vc_thread": 200},
        "BRASS": {"vc_rough": 250, "vc_finish": 300, "f_rough": 0.25, "f_finish": 0.10, "doc": 2.5,
                  "vc_mill": 200, "fz": 0.05, "vc_face": 250, "fz_face": 0.12, "vc_drill": 50, "f_drill": 0.12, "vc_tap": 12, "vc_thread": 150},
        "CI":    {"vc_rough": 150, "vc_finish": 180, "f_rough": 0.25, "f_finish": 0.12, "doc": 2.0,
                  "vc_mill": 100, "fz": 0.06, "vc_face": 120, "fz_face": 0.12, "vc_drill": 22, "f_drill": 0.12, "vc_tap": 7, "vc_thread": 90},
    },
}


def load_shop(path: str | None = None) -> dict:
    cfg = copy.deepcopy(DEFAULT)
    if path:
        with open(path) as f:
            user = json.load(f)
        _merge(cfg, user)
    return cfg


def _merge(a, b):
    for k, v in b.items():
        if isinstance(v, dict) and isinstance(a.get(k), dict):
            _merge(a[k], v)
        else:
            a[k] = v


def rpm(vc: float, dia: float, max_rpm: float) -> int:
    return int(min(max_rpm, 1000.0 * vc / (math.pi * max(dia, 0.5))))


TAP_DRILL = {(3, 0.5): 2.5, (4, 0.7): 3.3, (5, 0.8): 4.2, (6, 1.0): 5.0, (8, 1.25): 6.8, (8, 1.0): 7.0,
             (10, 1.5): 8.5, (10, 1.25): 8.8, (10, 1.0): 9.0, (12, 1.75): 10.2, (12, 1.5): 10.5, (12, 1.25): 10.8,
             (14, 2.0): 12.0, (14, 1.5): 12.5, (16, 2.0): 14.0, (16, 1.5): 14.5, (18, 2.5): 15.5, (20, 2.5): 17.5,
             (20, 1.5): 18.5, (22, 2.5): 19.5, (24, 3.0): 21.0}


def tap_drill(major: float, pitch: float) -> float:
    """Standard tap drill (ISO metric, ~75 % thread)."""
    key = (int(round(major)), round(pitch, 2))
    if key in TAP_DRILL and abs(major - round(major)) < 1e-6:
        return TAP_DRILL[key]
    return round(math.floor((major - pitch) * 10 + 0.5) / 10, 1)
