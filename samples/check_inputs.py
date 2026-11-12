"""Reads every sample in every input format and compares the part spec with the
DXF ground truth. Usage: python samples/check_inputs.py"""
import glob
import json
import os
import sys
import time
from dataclasses import asdict

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from drawcam.ingest.load import load_drawing  # noqa: E402
from drawcam.recognize.pipeline import recognize  # noqa: E402
from drawcam.spec import blocking  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))


def key(spec):
    s = asdict(spec)
    if s["machine"] == "lathe":
        return {k: s[k] for k in ("machine", "material", "profile", "threads", "grooves", "bore")}
    holes = sorted((h["kind"], h["dia"], h["x"], h["y"]) for h in s["holes"])
    pockets = sorted((p["shape"], p["depth"], p["w"], p["h"], p["r"], p["corner_r"]) for p in s["pockets"])
    return {"machine": s["machine"], "material": s["material"], "part_z": s["part_z"], "holes": holes, "pockets": pockets}


def main():
    ok = tot = safe = 0
    for name in ("shaft", "bush", "plate"):
        truth = key(recognize(load_drawing(os.path.join(HERE, name + ".dxf"))))
        files = [os.path.join(HERE, name + e) for e in (".pdf", "_scan.png")] + sorted(glob.glob(os.path.join(HERE, "raster", name + "_*")))
        for f in files:
            if not os.path.exists(f):
                continue
            tot += 1
            t = time.time()
            blocked = False
            try:
                spec = recognize(load_drawing(f))
                got = key(spec)
                diff = {k: (truth[k], got[k]) for k in truth if truth[k] != got[k]}
                blocked = bool(blocking(spec))
            except ValueError as ex:
                diff, blocked = {"refused": str(ex)[:100]}, True
            except Exception as ex:
                diff = {"error": str(ex)[:120]}
            ok += not diff
            safe += bool(diff) and blocked
            status = "OK     " if not diff else ("BLOCKED" if blocked else "DIFF   ")
            print(f"{status} {os.path.relpath(f, HERE):34s} {time.time() - t:5.1f}s",
                  "" if not diff else json.dumps(diff, default=str)[:300])
    print(f"\n{ok}/{tot} match the DXF exactly; {safe} more differ but G-code is blocked until a person fixes them;"
          f" {tot - ok - safe} differ without a block")
    return ok + safe == tot


if __name__ == "__main__":
    sys.exit(0 if main() else 1)
