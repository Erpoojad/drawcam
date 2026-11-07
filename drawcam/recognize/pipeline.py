from __future__ import annotations

from ..ingest.drawing import Drawing
from ..spec import ReviewItem
from .common import calibrate, classify_machine
from .milling import recognize_milling
from .turning import recognize_turning


def recognize(d: Drawing, machine: str | None = None):
    pre = []
    if machine is None:
        machine, conf, how = classify_machine(d)
        if conf < 0.8:
            pre.append(ReviewItem("machine", f"machine type '{machine}' guessed with {conf:.0%} confidence ({how}); pass --machine to force"))
    calibrate(d, pre, machine)
    spec = recognize_turning(d) if machine == "lathe" else recognize_milling(d)
    items, seen = [], set()
    for r in pre + [ReviewItem("reader", n) for n in d.notes if "skipped" in n] + spec.review:
        if (r.field, r.message) not in seen:
            seen.add((r.field, r.message))
            items.append(r)
    spec.review = items
    return spec
