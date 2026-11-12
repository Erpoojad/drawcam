"""Web app smoke tests (Flask test client, no server needed)."""
import html
import json
import os
import re

import pytest

S = os.path.join(os.path.dirname(__file__), "..", "samples")


@pytest.fixture()
def client(tmp_path, monkeypatch):
    import drawcam.web.app as web
    monkeypatch.setattr(web, "JOBS", str(tmp_path))
    web.app.config["TESTING"] = True
    return web.app.test_client()


def _spec(client, job_url):
    page = client.get(job_url).get_data(as_text=True)
    return html.unescape(re.search(r"<textarea[^>]*>(.*?)</textarea>", page, re.S).group(1))


@pytest.mark.parametrize("sample,progs", [("shaft", 2), ("bush", 1), ("plate", 1)])
def test_sample_to_gcode(client, sample, progs):
    r = client.post("/read", data={"sample": sample})
    assert r.status_code == 302 and "/job/" in r.location
    job = r.location
    r = client.post(job + "/gcode", data={"spec": _spec(client, job)})
    assert r.location.endswith("#programs")
    page = client.get(job).get_data(as_text=True)
    assert len(re.findall(r"<h2>O\d+_OP\d\.nc", page)) == progs
    z = client.get(job + "/zip")
    assert z.status_code == 200 and z.data[:2] == b"PK"


def test_upload_dxf(client):
    with open(os.path.join(S, "plate.dxf"), "rb") as f:
        r = client.post("/read", data={"drawing": (f, "my plate.dxf"), "machine": "mill"}, content_type="multipart/form-data")
    assert "/job/" in r.location
    assert "PLATE-12" in client.get(r.location).get_data(as_text=True)


def test_block_refuses_and_bad_input(client):
    job = client.post("/read", data={"sample": "plate"}).location
    spec = json.loads(_spec(client, job))
    spec["review"].append({"field": "material", "message": "x", "severity": "block"})
    r = client.post(job + "/gcode", data={"spec": json.dumps(spec)})
    assert "error=" in r.location
    assert "error=" in client.post(job + "/gcode", data={"spec": "{bad"}).location
    assert "error=" in client.post("/read", data={}).location
    assert client.get("/job/../../etc/passwd").status_code == 404
    assert client.get(job + "/file/..%2Fspec.json").status_code == 404


@pytest.mark.parametrize("fname,mime", [("bush_photo.jpg", "image/jpeg"), ("bush_photo_sideways.jpeg", "image/jpeg"),
                                         ("bush_scanned.pdf", "application/pdf"), ("bush_screenshot.png", "image/png")])
def test_upload_photo_png_jpeg_pdf(client, fname, mime):
    import sys
    sys.path.insert(0, S)
    import make_samples
    p = os.path.join(S, "raster", fname)
    if not os.path.exists(p):
        make_samples.make_raster_set(S)
    with open(p, "rb") as f:
        r = client.post("/read", data={"drawing": (f, fname.upper(), mime)}, content_type="multipart/form-data")
    assert "/job/" in r.location, r.location
    assert "BUSH-07" in client.get(r.location).get_data(as_text=True)


def test_heic_and_garbage_rejected_nicely(client):
    import io
    r = client.post("/read", data={"drawing": (io.BytesIO(b"x" * 100), "IMG_1234.HEIC")}, content_type="multipart/form-data")
    assert "HEIC" in r.location
    r = client.post("/read", data={"drawing": (io.BytesIO(b"not an image"), "photo.jpg")}, content_type="multipart/form-data")
    assert "error=" in r.location and "Traceback" not in r.location
