"""drawcam web app: upload a drawing -> review/edit the part spec -> G-code.

Same two-step safety flow as the CLI: nothing is generated while a 'block'
review item is open. Jobs live in a temp folder and are deleted after 24 h
(Render's disk is not permanent anyway).

Run locally:  python -m drawcam.web.app            (http://localhost:8000)
Production:   gunicorn -b 0.0.0.0:$PORT drawcam.web.app:app
"""
from __future__ import annotations

import argparse
import io
import json
import os
import re
import shutil
import time
import traceback
import uuid
import zipfile
from dataclasses import asdict

from flask import Flask, abort, redirect, render_template, request, send_file, url_for

from ..cli import cmd_gcode, cmd_read
from ..spec import blocking, from_json

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SAMPLES = os.path.join(ROOT, "samples")
JOBS = os.environ.get("DRAWCAM_JOBS", "/tmp/drawcam-jobs")
SHOP = os.environ.get("DRAWCAM_SHOP")              # optional path to a shop config JSON
ALLOWED = {".dxf", ".pdf", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
KEEP_HOURS = 24
SAMPLE_FILES = {
    "shaft": ("shaft.dxf", "Stepped shaft, M20 thread, 2 setups (lathe)"),
    "bush": ("bush.pdf", "Flanged bush, Ø20 H7 bore (lathe, PDF)"),
    "plate": ("plate.pdf", "Plate with holes, taps, pockets (VMC, PDF)"),
}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024
os.makedirs(JOBS, exist_ok=True)


def _cleanup():
    now = time.time()
    for d in os.listdir(JOBS):
        p = os.path.join(JOBS, d)
        try:
            if now - os.path.getmtime(p) > KEEP_HOURS * 3600:
                shutil.rmtree(p, ignore_errors=True)
        except OSError:
            pass


def _job(job_id):
    if not re.fullmatch(r"[0-9a-f]{12}", job_id or ""):
        abort(404)
    d = os.path.join(JOBS, job_id)
    if not os.path.isdir(d):
        abort(404)
    return d


def _ns(**kw):
    base = dict(dpi=300, machine=None, shop=SHOP, force=False, no_log=True)
    base.update(kw)
    return argparse.Namespace(**base)


def _outputs(d):
    progs = sorted(f for f in os.listdir(d) if f.endswith(".nc"))
    out = []
    for p in progs:
        stem = p[:-3]
        code = open(os.path.join(d, p), newline="").read().replace("\r\n", "\n")
        png = stem + "_backplot.png"
        out.append({"name": p, "code": code, "lines": code.count("\n"),
                    "png": png if os.path.exists(os.path.join(d, png)) else None})
    report = open(os.path.join(d, "report.md")).read() if os.path.exists(os.path.join(d, "report.md")) else ""
    return out, report


def _meta(d):
    p = os.path.join(d, "meta.json")
    return json.load(open(p)) if os.path.exists(p) else {}


# ------------------------------------------------------------------ routes
@app.get("/")
def index():
    return render_template("index.html", samples=SAMPLE_FILES, error=request.args.get("error"))


@app.get("/healthz")
def healthz():
    return "ok"


@app.post("/read")
def read():
    _cleanup()
    job_id = uuid.uuid4().hex[:12]
    d = os.path.join(JOBS, job_id)
    os.makedirs(d)
    sample = request.form.get("sample")
    if sample:
        if sample not in SAMPLE_FILES:
            abort(400)
        fname = SAMPLE_FILES[sample][0]
        src = os.path.join(d, fname)
        shutil.copy(os.path.join(SAMPLES, fname), src)
    else:
        f = request.files.get("drawing")
        if not f or not f.filename:
            return redirect(url_for("index", error="Choose a drawing file first."))
        ext = os.path.splitext(f.filename)[1].lower()
        if ext == ".dwg":
            return redirect(url_for("index", error="DWG can't be read directly: save it as DXF in your CAD software (or use the free ODA File Converter)."))
        if ext not in ALLOWED:
            return redirect(url_for("index", error=f"Unsupported file type {ext}. Use DXF, PDF, PNG or JPG."))
        fname = "drawing" + ext
        src = os.path.join(d, fname)
        f.save(src)
    machine = request.form.get("machine") or None
    if machine not in (None, "lathe", "mill"):
        machine = None
    try:
        dpi = int(request.form.get("dpi") or 300)
    except ValueError:
        dpi = 300
    t0 = time.time()
    try:
        spec = cmd_read(_ns(drawing=src, out=d, machine=machine, dpi=max(150, min(dpi, 600))))
    except Exception as ex:
        shutil.rmtree(d, ignore_errors=True)
        app.logger.error("read failed: %s", traceback.format_exc())
        return redirect(url_for("index", error=f"Could not read this drawing: {ex}"))
    json.dump({"original": (sample and SAMPLE_FILES[sample][0]) or request.files["drawing"].filename,
               "seconds": round(time.time() - t0, 1), "machine": spec.machine},
              open(os.path.join(d, "meta.json"), "w"))
    return redirect(url_for("job", job_id=job_id))


@app.get("/job/<job_id>")
def job(job_id):
    d = _job(job_id)
    spec = from_json(os.path.join(d, "spec.json"))
    progs, report = _outputs(d)
    return render_template("job.html", job_id=job_id, spec=spec, meta=_meta(d),
                           spec_json=open(os.path.join(d, "spec.json")).read(),
                           blocks=blocking(spec), progs=progs, report=report,
                           error=request.args.get("error"))


@app.post("/job/<job_id>/gcode")
def gcode(job_id):
    d = _job(job_id)
    text = request.form.get("spec", "")
    try:
        json.loads(text)
        tmp = os.path.join(d, "spec_edit.json")
        open(tmp, "w").write(text)
        spec = from_json(tmp)
    except Exception as ex:
        return redirect(url_for("job", job_id=job_id, error=f"The spec is not valid JSON: {ex}") + "#spec")
    # cleared review items count as "checked by a person"
    open(os.path.join(d, "spec.json"), "w").write(text)
    for f in os.listdir(d):
        if f.endswith((".nc", "_backplot.png")) or f == "report.md":
            os.remove(os.path.join(d, f))
    if blocking(spec):
        return redirect(url_for("job", job_id=job_id, error="Fix or delete every BLOCK item in the review list (and in the spec's \"review\" list) before generating.") + "#spec")
    try:
        rc = cmd_gcode(_ns(spec=None, out=d), spec)
    except Exception as ex:
        app.logger.error("gcode failed: %s", traceback.format_exc())
        return redirect(url_for("job", job_id=job_id, error=f"G-code generation failed: {ex}") + "#spec")
    if rc == 2:
        return redirect(url_for("job", job_id=job_id, error=f"Unknown material '{spec.material}'. Use one of MS, EN8, EN24, SS304, AL, BRASS, CI.") + "#spec")
    return redirect(url_for("job", job_id=job_id) + "#programs")


@app.get("/job/<job_id>/file/<path:name>")
def file(job_id, name):
    d = _job(job_id)
    if "/" in name or name.startswith("."):
        abort(404)
    p = os.path.join(d, name)
    if not os.path.isfile(p):
        abort(404)
    return send_file(p, as_attachment=not name.endswith(".png"))


@app.get("/job/<job_id>/zip")
def zip_all(job_id):
    d = _job(job_id)
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for f in sorted(os.listdir(d)):
            if f.endswith((".nc", ".png", ".md")) or f == "spec.json":
                z.write(os.path.join(d, f), f)
    buf.seek(0)
    spec = from_json(os.path.join(d, "spec.json"))
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", spec.part_name) or "part"
    return send_file(buf, mimetype="application/zip", as_attachment=True, download_name=f"{name}_programs.zip")


@app.errorhandler(413)
def too_big(_):
    return redirect(url_for("index", error="File is larger than 20 MB. Scan at 300 dpi in black and white."))


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), debug=False)
