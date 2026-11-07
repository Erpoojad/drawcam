"""drawcam command line.

  drawcam read  drawing.(dxf|pdf|png)  -o out/          -> out/spec.json + review list
  drawcam gcode out/spec.json          -o out/          -> O-programs, backplots, report
  drawcam run   drawing                -o out/          -> both; stops before G-code if anything blocks

The two-step flow is deliberate: read, open spec.json, fix what the review list
says, then generate. G-code is never produced while a 'block' item is open
(unless --force, which writes the programs with the problems listed at the top).
"""
from __future__ import annotations

import argparse
import json
import os
import sys

from .cam.milling import generate_milling
from .cam.shop import load_shop
from .cam.turning import generate_turning
from .ingest.load import load_drawing
from .recognize.pipeline import recognize
from .spec import blocking, from_json, to_json
from .verify.sim import backplot, sim_lathe, sim_mill


def cmd_read(args):
    from dataclasses import asdict
    from .recognize.common import machine_features
    d = load_drawing(args.drawing, dpi=args.dpi)
    feats = machine_features(d)
    spec = recognize(d, args.machine)
    os.makedirs(args.out, exist_ok=True)
    p = os.path.join(args.out, "spec.json")
    to_json(spec, p)
    with open(os.path.join(args.out, "auto_spec.json"), "w") as fh:     # kept untouched for the learning log
        json.dump({"drawing": os.path.abspath(args.drawing), "source": d.source,
                   "features": {k: float(v) for k, v in feats.items()}, "auto": asdict(spec)}, fh, indent=1)
    print(f"[{spec.machine}] spec written: {p}")
    for r in spec.review:
        print(f"  {'BLOCK' if r.severity == 'block' else 'check'}  {r.field}: {r.message}")
    return spec


def cmd_gcode(args, spec=None):
    spec = spec or from_json(args.spec)
    cfg = load_shop(args.shop)
    if spec.material not in cfg["materials"]:
        print(f"unknown material {spec.material}; add it to the shop config", file=sys.stderr)
        return 2
    blk = blocking(spec)
    if blk and not args.force:
        print("NOT generating G-code. Fix these in spec.json, then run 'drawcam gcode':")
        for r in blk:
            print(f"  BLOCK {r.field}: {r.message}")
        return 1
    progs = generate_turning(spec, cfg) if spec.machine == "lathe" else generate_milling(spec, cfg)
    os.makedirs(args.out, exist_ok=True)
    report = [f"# {spec.part_name} ({spec.machine}, {spec.material})", ""]
    total_min = 0.0
    any_err = False
    for op, code in progs.items():
        num = code.splitlines()[1]
        fn = os.path.join(args.out, f"{num}_{op}.nc")
        with open(fn, "w", newline="\r\n") as fh:      # CRLF: DNC/USB transfer friendly
            fh.write(code)
        if spec.machine == "lathe":
            moves, issues, minutes = sim_lathe(code, spec, cfg, flipped=(op == "OP2"))
        else:
            moves, issues, minutes = sim_mill(code, spec, cfg)
        png = os.path.join(args.out, f"{num}_{op}_backplot.png")
        backplot(moves, spec, png, f"{spec.part_name} {op}  -  est. {minutes:.1f} min")
        total_min += minutes
        any_err |= any(i.severity == "error" for i in issues)
        report += [f"## {op}: {os.path.basename(fn)}", f"- lines: {len(code.splitlines())}",
                   f"- estimated cycle time: {minutes:.1f} min (excluding loading)"]
        report += [f"- G-code check {i!r}" for i in issues] or ["- G-code check: no rapids into material, feeds and RPM within limits"]
        report.append("")
        print(f"{fn}  ({minutes:.1f} min, {len(issues)} check issues)")
        for i in issues:
            print(f"   {i!r}")
    if not getattr(args, "no_log", False):
        from .ml.dataset import log_correction
        rec = log_correction(args.out, spec)
        if rec:
            print(f"logged for learning: {rec['n_fields_changed']} field(s) changed by hand")
    report += ["## Review items", ""] + [f"- [{r.severity}] {r.field}: {r.message}" for r in spec.review]
    report += ["", f"Total estimated machine time per part: {total_min:.1f} min"]
    with open(os.path.join(args.out, "report.md"), "w") as fh:
        fh.write("\n".join(report) + "\n")
    with open(os.path.join(args.out, "spec_used.json"), "w") as fh:
        from dataclasses import asdict
        json.dump(asdict(spec), fh, indent=2)
    return 3 if any_err else 0


def main(argv=None):
    ap = argparse.ArgumentParser(prog="drawcam")
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("read", "run"):
        p = sub.add_parser(name)
        p.add_argument("drawing")
        p.add_argument("-o", "--out", default="out")
        p.add_argument("--machine", choices=["lathe", "mill"])
        p.add_argument("--dpi", type=int, default=300)
        p.add_argument("--shop")
        p.add_argument("--force", action="store_true")
        p.add_argument("--no-log", action="store_true")
    p = sub.add_parser("gcode")
    p.add_argument("spec")
    p.add_argument("-o", "--out", default="out")
    p.add_argument("--shop")
    p.add_argument("--force", action="store_true")
    p.add_argument("--no-log", action="store_true")
    sub.add_parser("stats", help="how often each field had to be corrected")
    args = ap.parse_args(argv)
    if args.cmd == "stats":
        from .ml.dataset import summary
        summary()
        return 0
    if args.cmd == "read":
        cmd_read(args)
        return 0
    if args.cmd == "gcode":
        return cmd_gcode(args)
    spec = cmd_read(args)
    return cmd_gcode(args, spec)


if __name__ == "__main__":
    sys.exit(main())
