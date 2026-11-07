"""NEXT STEP, NOT YET WIRED IN: a detector for drawing annotations on scans.

Why: on real scans the weakest link is finding and reading dimension text,
GD&T frames and surface-finish marks (Tesseract mis-reads Ø, ±, and anything
rotated). A small object detector that boxes these regions, followed by OCR
on each crop, is the standard fix. It needs YOUR labelled drawings - roughly
200-400 pages of the kind your customers actually send - so it ships as a
training script, not a trained model.

Labelling: use Label Studio or CVAT, export "YOLO" format:
    dataset/images/{train,val}/*.png
    dataset/labels/{train,val}/*.txt     (class cx cy w h, normalised)
    dataset/data.yaml                    (paths + names below)

Install: pip install ultralytics   (CPU training works for this size; a GPU is faster)

    python -m drawcam.ml.symbol_detector train dataset/data.yaml
    python -m drawcam.ml.symbol_detector detect drawing.png
"""
from __future__ import annotations

import sys

CLASSES = [
    "linear_dim",        # 25, 25±0.1
    "diameter_dim",      # Ø25 h7
    "radius_dim",        # R5
    "thread_callout",    # M10x1.5, 4X M8 DP 15
    "hole_callout",      # 4X Ø8.5 THRU, Ø6.8 DP 15
    "gdt_frame",         # feature control frame
    "datum",             # datum triangle + letter
    "surface_finish",    # Ra 1.6 tick
    "chamfer",           # C1, 1x45°
    "title_block",
]


def train(data_yaml: str, epochs: int = 120, imgsz: int = 1280, model: str = "yolov8n.pt"):
    from ultralytics import YOLO
    m = YOLO(model)
    m.train(data=data_yaml, epochs=epochs, imgsz=imgsz, batch=4, patience=30,
            degrees=0, fliplr=0.0, mosaic=0.5, hsv_v=0.3)       # drawings must not be mirrored
    return m


def detect(image: str, weights: str = "runs/detect/train/weights/best.pt", conf: float = 0.35):
    """Returns [(class_name, conf, x0, y0, x1, y1)] in pixels; crop these and OCR each with psm 7."""
    from ultralytics import YOLO
    res = YOLO(weights)(image, conf=conf, imgsz=1280)[0]
    out = []
    for b in res.boxes:
        x0, y0, x1, y1 = b.xyxy[0].tolist()
        out.append((res.names[int(b.cls)], float(b.conf), x0, y0, x1, y1))
    return out


if __name__ == "__main__":
    if len(sys.argv) >= 3 and sys.argv[1] == "train":
        train(sys.argv[2])
    elif len(sys.argv) >= 3 and sys.argv[1] == "detect":
        for r in detect(sys.argv[2]):
            print(r)
    else:
        print(__doc__)
