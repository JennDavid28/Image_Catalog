# Image Catalog (Assignment Question 1)

A local image catalog for object detection, editable metadata, manual corrections, and composites made from existing catalog assets. This repository implements Question 1 only; it does not implement the separate duplicate-question assignment.

## Features

- Upload and catalog images with dimensions, format, hash, detection status, route, timing, and provenance.
- Detect the assignment's illustrated shapes, colors, fruit, and people with the classical OpenCV pipeline. Optional neural backends support photographs and open-vocabulary prompts.
- Add, edit, remove, restore, confirm, and revert detected objects. Corrections persist until detection is explicitly rerun.
- Compose new catalog entries from detected or manually corrected catalog cut-outs. Missing requested assets are reported instead of being silently replaced.
- Optional hybrid and AI-generated image modes use the configured image generator; generated outputs are added to the catalog and detection runs on them.
- Edit images with crop, resize, rotate, flip, color adjustments, object operations, background blur, solid-color background replacement, and optional AI cut-out.
- Trash and restore catalog images and objects.

## Requirements and setup

Python 3.10 or newer is recommended. Install the base dependencies and generate the assignment test images:

```bash
python -m venv .venv
# Activate the environment, then:
pip install -r requirements.txt
python scripts/make_testdata.py
python scripts/fetch_assets.py   # optional YOLOX weights for photograph detection
uvicorn app.api:app --port 8000
```

Open http://localhost:8000 and choose **Load sample catalog**. The catalog is stored under `data/store/` by default. To use a separate store, create the app with a custom data directory in Python.

Optional model/API credentials belong in environment variables (or a local `.env` file loaded by your runtime). `.env` is ignored by Git. Never commit credentials. The Gemini planner uses `GEMINI_API_KEY` (or `GOOGLE_API_KEY`) and defaults to `gemini-2.5-flash`. AI generation and optional segmentation have their own provider/dependency requirements; the offline fallback behavior is described by their modules and configured at runtime. There is no API endpoint for saving secrets.

Optional open-vocabulary detection:

```bash
pip install -r requirements-openvocab.txt
# YOLO-World is enabled when ultralytics and its configured weights are available.
# OWLv2 can be enabled with torch + transformers; first use may need model download access.
```

## Run and evaluate

```bash
pytest
python scripts/ui_smoke.py       # optional; requires Playwright
python scripts/demo.py           # regenerate examples and evidence
python -m app.evaluate           # detector metrics on the generated ground-truth set
```

The benchmark compares hybrid, classical, YOLO-World, and OWLv2. Open-vocabulary models receive a fixed generic vocabulary, never labels taken from the ground truth. A model without its required package or weights is marked unavailable; its zero metrics are not a measured performance result. This synthetic assignment dataset is small and is not a general measure of real-world detection quality.

## Architecture

| Module | Responsibility |
|---|---|
| `app/api.py` | FastAPI routes and request validation |
| `app/catalog.py` | SQLite/filesystem storage, metadata provenance, corrections, audit events, image operations |
| `app/detector.py` | Detection routing and merging of classical/neural proposals |
| `app/neural.py` | YOLOX-nano COCO detector via OpenCV DNN |
| `app/yoloworld.py` | Optional Ultralytics YOLO-World open-vocabulary detector |
| `app/openvocab.py` | Optional OWLv2 text-guided detector |
| `app/compose.py` | Prompt parsing, asset resolution, layout, rendering, and composite metadata |
| `app/planner.py` | Gemini-assisted planning with deterministic local fallback |
| `app/generator.py` | AI image generation provider and local fallback |
| `app/imageops.py` | Pixel-level editing and background operations |
| `app/evaluate.py` | Ground-truth metrics and model comparison |
| `static/index.html` | Browser UI (vanilla JavaScript and SVG overlays) |

## Detection and correction behavior

Flat illustrations use color and geometry rules to find regions and classify common assignment objects. Complex images use YOLOX proposals when weights are installed; optional YOLO-World and OWLv2 can add open-vocabulary proposals. Normal detection on new images now invokes YOLO-World with a fixed common-label vocabulary when the backend is available. User-supplied labels can be added during explicit re-detection. If a backend is unavailable, the applicable classical/YOLOX routes continue where possible and status reports capability limitations.

Object labels, colors, confidence, and bounding boxes retain per-field provenance (`auto` or `user`). The original automatic values are retained for reverting. `preserve_manual` updates only untouched automatic detections; manual additions, edits, confirmations, and removals remain in effect. `replace_all` explicitly replaces detections, including corrections. New generated images are also detected and receive a final detection status.

## Composite behavior

The default **Assets** mode parses the instruction, resolves requested objects to original catalog images, lays out cut-outs, and saves a new composite entry with provenance and object boxes. It refuses to substitute approximations for missing catalog assets. **Hybrid** and **Pure AI** are separate explicit modes and require a working generator; those modes can synthesize pixels by design. The LLM planner is optional and does not grant permission to invent catalog assets in Assets mode. Without a Gemini key, local parsing/resolution is used.

## Security and scope

This is a single-user local application with no authentication, multi-user isolation, or network deployment security. Do not expose it to an untrusted network. Authentication and the seeded demo accounts have been removed; no login screen or authentication API is provided. API settings do not accept or persist API keys.

## Known limitations

- The generated assignment test images are synthetic and easier than varied real artwork or photography.
- Cartoon boy/girl classification is a visual convention and should be reviewed.
- Optional model performance depends on installed packages, weight availability, hardware, and the supplied vocabulary.
- Mask quality and recolor/erase results can be imperfect, especially on complex photography.
- The local fallback generator is not equivalent to a hosted image-generation model.
- Trash has no automatic expiry; permanently purged files cannot be restored.
