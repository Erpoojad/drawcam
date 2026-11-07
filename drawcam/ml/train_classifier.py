"""Trains the lathe-vs-mill drawing classifier.

Data: randomly generated turned and prismatic parts, each rendered as DXF and
vector PDF (and a share as noisy scans), plus any real drawings you have
corrected (records in the dataset log written by `drawcam gcode`). Features
are the geometric/text features in recognize.common.machine_features.

    python -m drawcam.ml.train_classifier --n 300 --scans 40

The rules in classify_machine() still work without a model; the model mainly
helps on scans and unusual drawings, and it improves as real corrected
drawings are added.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import tempfile

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "samples"))

from make_samples import Part, revolve_side, to_dxf, to_pdf, to_scan  # noqa: E402

from ..ingest.load import load_drawing  # noqa: E402
from ..recognize.common import FEATURES, MODEL_PATH, machine_features, rule_classify  # noqa: E402


def rand_turned(rng: random.Random) -> Part:
    p = Part("t")
    n = rng.randint(1, 5)
    prof, total = [], 0
    for _ in range(n):
        L = rng.choice([5, 8, 10, 12, 15, 20, 25, 30, 40])
        D = rng.choice([10, 12, 16, 20, 25, 30, 35, 40, 50, 60, 80])
        prof.append((L, D))
        total += L
    revolve_side(p, total, 0, prof)
    if rng.random() < 0.4:                      # bore shown with hidden lines
        bd = min(D for _, D in prof) * rng.uniform(0.3, 0.6)
        p.L(0, bd / 2, total, bd / 2, "hidden")
        p.L(0, -bd / 2, total, -bd / 2, "hidden")
    if rng.random() < 0.6:                      # end view
        cx = total + 20 + max(D for _, D in prof) / 2
        for _, D in prof:
            p.C(cx, 0, D / 2)
        p.L(cx - 40, 0, cx + 40, 0, "center")
    for L, D in prof:
        p.T(f"Ø{D}", rng.uniform(0, total), D / 2 + 5)
    p.T(str(total), total / 2, -max(D for _, D in prof) / 2 - 10)
    if rng.random() < 0.3:
        p.T(f"M{prof[0][1]}x1.5", total - 5, prof[0][1] / 2 + 10)
    p.title = {"MATERIAL": rng.choice(["EN8", "MS", "SS304"])}
    return p


def rand_prismatic(rng: random.Random) -> Part:
    p = Part("m")
    W, H, T = rng.choice([40, 60, 80, 100, 120, 150]), rng.choice([30, 40, 60, 80, 100]), rng.choice([8, 10, 12, 15, 20, 25])
    round_flange = rng.random() < 0.2
    if round_flange:                             # flange top view: circle outline + PCD holes
        R = W / 2
        p.C(R, R, R)
        k = rng.choice([3, 4, 6, 8])
        for i in range(k):
            a = 2 * math.pi * i / k
            p.C(R + 0.7 * R * math.cos(a), R + 0.7 * R * math.sin(a), rng.choice([3.25, 4.25, 5.5]))
        p.C(R, R, 0.3 * R)
        p.L(-5, R, 2 * R + 5, R, "center")
        p.L(R, -5, R, 2 * R + 5, "center")
        p.T(f"{k}X Ø9 THRU ON PCD {1.4 * R:g}", R, 2 * R + 6)
        H = W
    else:
        p.L(0, 0, W, 0); p.L(W, 0, W, H); p.L(W, H, 0, H); p.L(0, H, 0, 0)
        for _ in range(rng.randint(0, 6)):
            d = rng.choice([4.2, 5.5, 6.8, 8.5, 10, 12])
            p.C(rng.uniform(d, W - d), rng.uniform(d, H - d), d / 2)
        if rng.random() < 0.5:
            w, h = W * rng.uniform(0.2, 0.5), H * rng.uniform(0.2, 0.5)
            x0, y0 = rng.uniform(5, W - w - 5), rng.uniform(5, H - h - 5)
            p.L(x0, y0, x0 + w, y0); p.L(x0 + w, y0, x0 + w, y0 + h); p.L(x0 + w, y0 + h, x0, y0 + h); p.L(x0, y0 + h, x0, y0)
            p.T("POCKET DP 5", x0 + w / 2, y0 + h / 2)
        if rng.random() < 0.5:                    # centre line through a hole row: the classic trap
            yy = rng.uniform(10, H - 10)
            p.L(-5, yy, W + 5, yy, "center")
            for xx in (W * 0.25, W * 0.75):
                p.C(xx, yy, 4.25)
        p.T(str(W), W / 2, H + 8)
        p.T(str(H), -10, H / 2)
    if rng.random() < 0.7:                       # side view
        p.L(0, -T - 20, W, -T - 20); p.L(W, -T - 20, W, -20); p.L(W, -20, 0, -20); p.L(0, -20, 0, -T - 20)
        p.T(str(T), W + 8, -20 - T / 2)
    p.title = {"MATERIAL": rng.choice(["AL 6061", "MS", "EN8"]), "THK": str(T)}
    return p


def build(n: int, n_scans: int, seed: int = 0):
    rng = random.Random(seed)
    X, y, groups = [], [], []
    tmp = tempfile.mkdtemp(prefix="drawcam_synth_")
    for i in range(n):
        label = "lathe" if i % 2 == 0 else "mill"
        part = rand_turned(rng) if label == "lathe" else rand_prismatic(rng)
        base = os.path.join(tmp, f"p{i}")
        to_dxf(part, base + ".dxf")
        to_pdf(part, base + ".pdf", rng.choice([1.0, 1.0, 2.0, 0.5]))
        files = [base + ".dxf", base + ".pdf"]
        if i < n_scans:
            to_scan(base + ".pdf", base + ".png", seed=i)
            files.append(base + ".png")
        for fp in files:
            try:
                f = machine_features(load_drawing(fp))
            except Exception as ex:
                print("skip", fp, ex)
                continue
            X.append([f[k] for k in FEATURES])
            y.append(label)
            groups.append(i)
        if (i + 1) % 50 == 0:
            print(f"  {i + 1}/{n} parts")
    return np.array(X, float), np.array(y), np.array(groups)


def load_real(path):
    """Corrected real drawings logged by `drawcam gcode` (see ml/dataset.py)."""
    X, y = [], []
    if path and os.path.exists(path):
        for line in open(path):
            r = json.loads(line)
            if "features" in r and r.get("machine"):
                X.append([r["features"][k] for k in FEATURES])
                y.append(r["machine"])
    return np.array(X, float).reshape(-1, len(FEATURES)), np.array(y)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--scans", type=int, default=40)
    ap.add_argument("--real", default=os.path.expanduser("~/.drawcam/dataset.jsonl"))
    ap.add_argument("--out", default=MODEL_PATH)
    a = ap.parse_args(argv)
    print("generating synthetic drawings...")
    X, y, groups = build(a.n, a.scans)
    Xr, yr = load_real(a.real)
    if len(yr):
        print(f"adding {len(yr)} real corrected drawings (weighted x5)")
        X = np.vstack([X] + [Xr] * 5)
        y = np.concatenate([y] + [yr] * 5)
        groups = np.concatenate([groups] + [np.arange(len(yr)) + 10 ** 6] * 5)
    # split by PART, not by file: the DXF, PDF and scan of one part must not sit on
    # both sides of the split, or the test score is flattering
    from sklearn.model_selection import GroupShuffleSplit
    tr, te = next(GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=1).split(X, y, groups))
    Xtr, Xte, ytr, yte = X[tr], X[te], y[tr], y[te]
    clf = RandomForestClassifier(n_estimators=300, max_depth=8, min_samples_leaf=2, random_state=0)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    rule = np.array([rule_classify(dict(zip(FEATURES, row)))[0] for row in Xte])
    print(f"hand-written rules accuracy on the same test set: {(rule == yte).mean():.3f}")
    print(f"random forest accuracy:                          {(pred == yte).mean():.3f}")
    print(classification_report(yte, pred, digits=3))
    print("confusion (rows true lathe/mill):\n", confusion_matrix(yte, pred, labels=["lathe", "mill"]))
    imp = sorted(zip(clf.feature_importances_, FEATURES), reverse=True)
    print("feature importance:", ", ".join(f"{n}={v:.2f}" for v, n in imp))
    clf.fit(X, y)                                # final model on everything
    joblib.dump(clf, a.out)
    print("saved", os.path.abspath(a.out))
    # independent check: the hand-drawn sample drawings were not made by this generator
    sdir = os.path.join(HERE, "..", "..", "samples")
    truth = {"shaft": "lathe", "bush": "lathe", "plate": "mill"}
    ok = tot = 0
    for name, lab in truth.items():
        for ext in (".dxf", ".pdf", "_scan.png"):
            fp = os.path.join(sdir, name + ext)
            if os.path.exists(fp):
                f = machine_features(load_drawing(fp))
                pr = clf.predict_proba(np.array([[f[k] for k in FEATURES]]))[0]
                guess = clf.classes_[pr.argmax()]
                ok += guess == lab
                tot += 1
                print(f"  sample {name + ext:16s} -> {guess} ({pr.max():.0%})")
    if tot:
        print(f"independent sample drawings: {ok}/{tot} correct")


if __name__ == "__main__":
    main()
