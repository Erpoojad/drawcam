"""Single entry point: any drawing file -> Drawing."""
from __future__ import annotations

import os

from .drawing import Drawing


IMAGE_EXT = (".png", ".jpg", ".jpeg", ".jfif", ".tif", ".tiff", ".bmp", ".webp", ".gif")


def load_drawing(path: str, dpi: int = 300) -> Drawing:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".dxf":
        from .dxf_reader import read_dxf
        return read_dxf(path)
    if ext == ".dwg":
        raise ValueError("DWG is a closed format: convert to DXF first (ODA File Converter is free).")
    if ext == ".pdf":
        import pymupdf
        from .pdf_reader import is_vector, read_pdf
        try:
            doc = pymupdf.open(path)
        except Exception:
            raise ValueError("this PDF could not be opened (damaged or password-protected)") from None
        if doc.needs_pass:
            raise ValueError("this PDF is password-protected: remove the password and upload again")
        if is_vector(path):
            return read_pdf(path)
        from .scan_reader import read_scan, pdf_page_to_image
        return read_scan(pdf_page_to_image(path, dpi), dpi)
    if ext in IMAGE_EXT:
        from .scan_reader import read_scan
        return read_scan(path, dpi)
    if ext in (".heic", ".heif"):
        raise ValueError("iPhone HEIC photos aren't supported: share the photo as JPG (or set Camera > Formats > Most Compatible)")
    raise ValueError(f"unsupported drawing type: {ext}")
