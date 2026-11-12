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
