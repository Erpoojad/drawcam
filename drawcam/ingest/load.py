"""Single entry point: any drawing file -> Drawing."""
from __future__ import annotations

import os

from .drawing import Drawing


def load_drawing(path: str, dpi: int = 300) -> Drawing:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".dxf":
        from .dxf_reader import read_dxf
        return read_dxf(path)
    if ext == ".dwg":
        raise ValueError("DWG is a closed format: convert to DXF first (ODA File Converter is free).")
    if ext == ".pdf":
        from .pdf_reader import is_vector, read_pdf
        if is_vector(path):
            return read_pdf(path)
        from .scan_reader import read_scan, pdf_page_to_image
        return read_scan(pdf_page_to_image(path, dpi), dpi)
    if ext in (".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"):
        from .scan_reader import read_scan
        return read_scan(path, dpi)
    raise ValueError(f"unsupported drawing type: {ext}")
