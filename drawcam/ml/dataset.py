"""Learning from corrections.

Every time a spec is read (auto) and later turned into G-code (after a human
fixed it), the pair is logged. That log is the real training data for this
project: it says exactly where the reader goes wrong on YOUR customers'
drawings, and it feeds train_classifier.py.

Record (one JSON object per line):
  time, drawing, source, machine, features, n_fields_changed, changes[], auto, final
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict

DEFAULT_LOG = os.path.expanduser("~/.drawcam/dataset.jsonl")


def _flat(d, prefix=""):
    out = {}
    if isinstance(d, dict):
        for k, v in d.items():
            if k == "review":
                continue
            out.update(_flat(v, f"{prefix}{k}."))
    elif isinstance(d, list):
        for i, v in enumerate(d):
            out.update(_flat(v, f"{prefix}{i}."))
    else:
        out[prefix[:-1]] = d
    return out


def diff(auto: dict, final: dict) -> list:
    a, b = _flat(auto), _flat(final)
    ch = []
    for k in sorted(set(a) | set(b)):
        va, vb = a.get(k), b.get(k)
        if isinstance(va, float) and isinstance(vb, float) and abs(va - vb) < 1e-6:
            continue
        if va != vb:
            ch.append({"field": k, "auto": va, "final": vb})
    return ch


def log_correction(out_dir: str, final_spec, path: str = DEFAULT_LOG):
    meta_p = os.path.join(out_dir, "auto_spec.json")
    if not os.path.exists(meta_p):
        return None
    meta = json.load(open(meta_p))
    final = asdict(final_spec)
    changes = diff(meta["auto"], final)
    rec = {"time": time.strftime("%Y-%m-%d %H:%M:%S"), "drawing": meta.get("drawing"), "source": meta.get("source"),
           "machine": final_spec.machine, "features": meta.get("features"), "n_fields_changed": len(changes),
           "changes": changes, "auto": meta["auto"], "final": final}
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(rec) + "\n")
    return rec


def summary(path: str = DEFAULT_LOG):
    """Which fields get corrected most often = where to improve the reader next."""
    from collections import Counter
    if not os.path.exists(path):
        print("no records yet")
        return
    recs = [json.loads(l) for l in open(path)]
    by_src = Counter(r["source"] for r in recs)
    clean = Counter(r["source"] for r in recs if r["n_fields_changed"] == 0)
    print(f"{len(recs)} drawings logged")
    for s, n in by_src.items():
        print(f"  {s:5s}: {clean[s]}/{n} needed no correction")
    fields = Counter()
    for r in recs:
        for c in r["changes"]:
            fields[".".join(p for p in c["field"].split(".") if not p.isdigit())] += 1
    print("most corrected fields:")
    for f, n in fields.most_common(10):
        print(f"  {n:4d}  {f}")


if __name__ == "__main__":
    summary()
