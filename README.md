# drawcam — engineering drawing → CNC lathe / VMC program

Reads a 2D engineering drawing (DXF, CAD-exported PDF, or a scanned/photographed sheet),
recognises the part, and writes Fanuc-style G-code for a CNC turning centre or a VMC,
with a backplot, a safety check of the generated code, and a cycle-time estimate.

**It is a programming assistant, not an autopilot.** Every program must be proved out
on the machine (single block, rapid and feed override low, dry run above the part) before
cutting. The tool is built so a human checks the part before any code exists.

---

## Why it is not "one ML model"

A single network that takes a drawing image and emits G-code does not work, and no
commercial CAM product does it either:

* There is no training data of (drawing, correct G-code) pairs, and a shop would need
  tens of thousands of them.
* G-code has zero tolerance for "approximately right". One wrong digit is a crash.
* Most of the job — toolpaths, canned cycles, feeds and speeds — is geometry and
  engineering rules that can be computed exactly. Learning them would only add error.

So ML is used only where the problem is genuinely perception, and everything after the
part is understood is deterministic:

```
 drawing ──► INGEST ───────────► RECOGNISE ─────────► spec.json ──► CAM ──────────► .nc
 (dxf/pdf/   lines, circles,      lathe or mill? (ML)   human        Fanuc cycles     │
  scan)      text (OCR = ML)      profile / holes /     reviews      G71 G70 G76      ▼
                                  pockets / threads /   & fixes      G83 G84 G85   VERIFY
                                  tolerances                                       re-reads the
                                                                                   G-code: rapids
                                                                                   into material,
                                                                                   RPM, feeds,
                                                                                   backplot, time
```

| Stage | Method | ML? |
|---|---|---|
| Read DXF | exact geometry (ezdxf) | no |
| Read vector PDF | lines, line weights, dash patterns, text (PyMuPDF) | no |
| Read scan | stroke-width separation, contour circle fitting, region extraction (OpenCV) | classical CV |
| Read text on scans | Tesseract LSTM OCR + drawing-specific corrections (Ø read as @/O/0, `Cl` → C1) | **yes (pretrained)** |
| Lathe or mill? | random forest on 10 drawing features, rule fallback | **yes (trained here)** |
| Annotation detector for scans | YOLO training script — needs your labelled drawings | **next step** |
| Learn from corrections | every hand-fixed spec is logged and diffed | **data for retraining** |
| Toolpaths, cycles, feeds | rules + your shop config | no, on purpose |

---

## Quick start

```bash
pip install -r requirements.txt          # plus Tesseract OCR on the system
python samples/make_samples.py            # builds 3 test parts x 3 formats (scans are regenerated here, not shipped)
python -m drawcam.cli run samples/shaft.dxf -o out/shaft
```

The real workflow is two steps:

```bash
python -m drawcam.cli read  customer_drawing.pdf -o jobs/1234     # writes spec.json + review list
#   open jobs/1234/spec.json, fix whatever the review list says
python -m drawcam.cli gcode jobs/1234/spec.json -o jobs/1234 --shop my_shop.json
```

`gcode` **refuses** while any review item has severity `block` (unknown material,
missing thickness, a thread with no cylinder, a pocket with no depth…). `--force`
exists but writes the problems into the program header.

Output per job: `O1001_OP1.nc` (and `O1002_OP2.nc` if the part needs flipping),
a backplot PNG per program, `report.md` with check results, review items and cycle time.

### Make it your shop's

Copy `shop_config.example.json` and set turret positions, max RPM, travels and the
cutting data your operators actually trust. **The defaults are textbook starting values
for carbide turning/milling and HSS drills/taps, not proven numbers for your machines.**

---

## Web app (Render)

The same workflow runs in the browser: upload a drawing, read the review list, edit
the part spec, generate, download the programs as a zip.

```bash
python -m drawcam.web.app          # local: http://localhost:8000
```

Deployed with the `Dockerfile` (it installs Tesseract OCR, which a plain Python
runtime can't). `render.yaml` describes the service. On the free plan the server sleeps
after 15 minutes idle (first request then takes ~1 minute) and uploaded jobs are
deleted on restart or after 24 hours, so download your programs.

Anyone with the link can use it. Don't upload drawings a customer considers
confidential to a public link: run it locally for those.

## What it handles

**Lathe (Fanuc 0i-T style)** — facing; stepped, tapered and chamfered OD profiles
(G71 rough, G70 finish with G42); edge-break chamfers on every outward shoulder; relief
grooves (G75); metric external threads (G76, runs out into the relief groove); drilled and
bored through/blind bores (G74, internal G71/G70); part-off (G75). Profiles that get smaller
again after the largest diameter are split into OP1 + OP2 (flip the part). Diameters are
programmed to the **middle of the tolerance band**, including ISO fits (H7, h6, g6, f7, k6, m6, p6).

**VMC (Fanuc 0i-M style)** — face milling; rectangular pockets with corner radii, round
pockets (helical entry, climb rings), free-form pockets (offset rings, ramp entry); outer
contour; spot drill; drilling (G81/G83 by depth); rigid tapping (M29 + G84) with standard tap
drills and depth limited so the drill never breaks through under a blind tap; reaming
(G85). Holes inside pockets start from the pocket floor. Tools are chosen per feature
(end mill ≤ 0.8 × corner radius, ≤ half a round pocket, etc.).

**Drawing conventions understood** — Ø ⌀ %%c, R, M10 / M20x1.5 (coarse pitch filled in),
`4X`, THRU, DP / DEPTH, ±, +a/-b, ISO fit letters, C1 / 1x45°, title-block MATERIAL,
SCALE, THK, PART NO, QTY; centre/hidden linetypes or layer names; 3/4-arc thread symbols.

---

## Input formats

| Input | How it's read | Reliability |
|---|---|---|
| DXF | exact geometry | best |
| PDF exported from CAD | vector lines and text | very good |
| PDF from a scanner (incl. ones with a hidden OCR text layer) | rendered, then read as an image | fair |
| PNG / JPG / JPEG / TIFF / WEBP scan | image pipeline | fair |
| Phone photo (JPG) | finds the sheet, removes perspective and shadows, uses EXIF rotation, then the image pipeline | fair, needs care |
| DWG / HEIC | rejected with instructions (save as DXF / JPG) | — |

For images the reader also: estimates the sheet size from line weights when the file has
no dpi, squares the image up on the drawing's border frame, picks the drawing scale by
voting across all dimensions (one OCR misread can't throw it off), and corrects X and Y
stretch separately. When it can't read something that matters (a depth, a groove, a
taper it isn't sure of), it **blocks** G-code instead of guessing.

`python samples/check_inputs.py` runs every sample in every format against the DXF
ground truth. Current result on 18 inputs (DXF-derived PDF, scan, photo, sideways photo,
scanned PDF, low-res screenshot for 3 parts): **12 exact, 4 blocked for a human to fix,
2 off by ≤ 0.015 mm** (tolerance text too small to read at 130 dpi).

## Results on the test drawings

Three parts (stepped EN8 shaft with M20x1.5 thread and relief groove needing two setups;
MS flanged bush with Ø20 H7 bore; Al plate with 4 through holes, 2× M8 blind taps,
Ø10 H7 reamed hole, 40×30 R5 pocket and Ø30 round pocket), each as DXF, vector PDF at a
different drawing scale, and a noisy 300 dpi scan:

* **9/9 drawings** recognised with the same part spec as the DXF ground truth
  (all diameters, steps, thread, groove, bore, every hole position, pocket sizes).
* Generated programs pass the G-code checker (no rapid into material, no feed move
  without F, RPM within limits) and the tests plant crashes to prove the checker catches them.
* Machine-type classifier: 100 % on a held-out synthetic set (split by part) and 9/9 on
  the independent sample drawings, with only 73–83 % confidence on the less typical
  ones — so it still flags them for a human.
* `python -m pytest -q` → 45 tests (pipeline, image inputs, web app).

Read these numbers for what they are: the samples were drawn cleanly, by a generator I
wrote, following ISO line weights. **Real customer drawings will do worse**, especially
scans, and the honest measure is the correction log on your own drawings (below).

---

## Known limits (in order of how much they will hurt)

1. **Scans and photos of real-world drawings.** Test photos here are simulated (perspective,
   shadow, JPEG, EXIF rotation) from clean drawings. Coffee stains, faded prints, skew, hand-written
   corrections, rotated dimension text and stamps will break the scan reader far more
   often than the clean samples suggest. Expect to fix scans by hand; insist on DXF/PDF
   from customers where you can.
2. **One view at a time.** The lathe reader uses the side view; the mill reader uses the
   top view plus thickness. Features only visible in a section or another view (cross
   holes, flats, keyways, slots in the side, undercuts on faces) are not recognised.
3. **No 3D.** No STEP/IGES, no 4th-axis, no 3D surfacing.
4. **Lathe**: external threads only on the OP1 side, no internal threads, no face
   grooves, no live tooling. The boring cycle assumes a plain bore.
5. **VMC**: counterbores/spotfaces are flagged, not machined; the outer contour assumes
   the part sits on parallels clear of the vise (or delete that section and finish the
   sides in a second setup). Pocket finish uses computed paths, not G41, so wear offsets
   can't trim pocket size.
6. **Controller dialect.** Written for Fanuc 0i. Siemens/Mitsubishi/Haas need a different
   post-processor (canned-cycle syntax differs). If you are not sure what your control
   is, check the screen bezel before running anything.

## The learning loop (where the real ML value comes from)

`read` saves the untouched automatic spec; `gcode` compares it with the corrected spec
you actually used and appends the difference to `~/.drawcam/dataset.jsonl`.

```bash
python -m drawcam.cli stats          # which fields you correct most, per input type
python -m drawcam.ml.train_classifier --n 300 --scans 40   # retrains, adds your real drawings x5
```

After 50–100 real jobs this log tells you exactly which part of the reader to improve next,
and becomes the training set for the scan annotation detector (`drawcam/ml/symbol_detector.py`).

## Project layout

```
drawcam/
  ingest/      dxf_reader, pdf_reader, scan_reader, dim_parser (callouts, ISO fits), load
  recognize/   common (views, calibration, features, classifier), turning, milling, pipeline
  cam/         shop (machines, tools, cutting data, tap drills), turning, milling, gcode
  verify/      sim (G-code read-back, checks, backplot, cycle time)
  ml/          train_classifier, dataset (correction log), symbol_detector (next step)
  web/         Flask app + templates (upload, review, generate)
  cli.py
Dockerfile, render.yaml, requirements-deploy.txt   deployment
samples/       make_samples.py + 9 test drawings
examples/      full output for every sample (programs, backplots, reports)
tests/         test_pipeline.py
```

## Prove-out checklist (print it, tape it to the machine)

1. Read the `CHECK:` lines at the top of the program.
2. Tools in the turret/magazine match the tool list; offsets and nose radius/tip set.
3. Work offset set (lathe Z0 = finished face; VMC X0Y0 = part corner, Z0 = finished top).
4. Graphics/dry-run on the control.
5. First part: single block, rapid 25 %, feed override low, hand on feed hold.
6. Measure, adjust wear offsets, then run.
