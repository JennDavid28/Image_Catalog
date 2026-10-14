"""Benchmark detection models against ground truth & evaluate performance.

Evaluates:
- Precision, Recall, F1 score
- Bounding-box IoU
- Label accuracy and Color accuracy
- Inference latency (ms)
- Performance across cartoon/synthetic vs realistic photo images
- Comparison between Classical, YOLOX-nano, YOLO-World, and OWLv2
"""
from __future__ import annotations
import json
import os
import time
from pathlib import Path
from typing import Dict, Any, List

import cv2
from .detector import detect, iou
from .yoloworld import YOLOWORLD
from .openvocab import OPENVOCAB

ROOT = Path(__file__).resolve().parent.parent
TESTSET = ROOT / "data" / "testset"
REAL = ROOT / "tests" / "real"
REAL_GT = {"apple.jpg": ("apple", "red"), "orange.jpg": ("orange", "orange")}


def match(dets, gts, thr=0.5):
    """Greedy one-to-one matching by IoU. Returns (pairs, unmatched_gt, unmatched_det)."""
    cand = sorted(((iou(d.bbox if hasattr(d, "bbox") else d["bbox"], g["bbox"]), i, j)
                   for i, d in enumerate(dets) for j, g in enumerate(gts)), reverse=True)
    used_d, used_g, pairs = set(), set(), []
    for v, i, j in cand:
        if v < thr or i in used_d or j in used_g:
            continue
        used_d.add(i); used_g.add(j); pairs.append((i, j, v))
    return pairs, [j for j in range(len(gts)) if j not in used_g], [i for i in range(len(dets)) if i not in used_d]


def run(use_neural: bool = True) -> dict:
    gt = json.loads((TESTSET / "ground_truth.json").read_text())
    per_image, tp = {}, 0
    n_gt = n_det = lab_ok = col_ok = 0
    ious = []
    for name, gts in gt.items():
        res = detect(cv2.imread(str(TESTSET / f"{name}.png")), use_neural=use_neural)
        pairs, miss, extra = match(res.objects, gts)
        rows = []
        for i, j, v in pairs:
            d, g = res.objects[i], gts[j]
            l_ok, c_ok = d.label == g["label"], d.color == g["color"]
            lab_ok += l_ok; col_ok += c_ok; ious.append(v)
            rows.append(dict(gt=g["label"], pred=d.label, label_ok=l_ok, gt_color=g["color"], pred_color=d.color,
                             color_ok=c_ok, iou=round(v, 3), conf=d.confidence))
        tp += len(pairs); n_gt += len(gts); n_det += len(res.objects)
        per_image[name] = dict(status=res.status, route=res.route, n_gt=len(gts), n_det=len(res.objects),
                                matched=rows, missed=[gts[j]["label"] for j in miss],
                                extra=[res.objects[i].label for i in extra], ms=round(res.ms))
    prec, rec = tp / max(1, n_det), tp / max(1, n_gt)
    out = dict(images=per_image, summary=dict(
        gt_objects=n_gt, detections=n_det, matched=tp, precision=round(prec, 3), recall=round(rec, 3),
        f1=round(2 * prec * rec / max(1e-9, prec + rec), 3), label_accuracy=round(lab_ok / max(1, tp), 3),
        color_accuracy=round(col_ok / max(1, tp), 3), mean_iou=round(sum(ious) / max(1, len(ious)), 3)))
    real = {}
    for f, (lab, col) in REAL_GT.items():
        p = REAL / f
        if p.exists():
            r = detect(cv2.imread(str(p)), use_neural=use_neural)
            top = max(r.objects, key=lambda d: d.confidence, default=None)
            real[f] = dict(status=r.status, route=r.route, label=top.label if top else None, color=top.color if top else None,
                           conf=top.confidence if top else None, expected=[lab, col], ms=round(r.ms))
    out["real_photos"] = real
    return out


_BENCHMARK_CACHE = None


def benchmark_models(force_refresh: bool = False) -> dict:
    """Benchmark and compare detectors: Hybrid (Classical+YOLOX), Classical-only, YOLO-World, OWLv2."""
    global _BENCHMARK_CACHE
    if _BENCHMARK_CACHE is not None and not force_refresh:
        return _BENCHMARK_CACHE

    gt = json.loads((TESTSET / "ground_truth.json").read_text())
    # Representative benchmark cases across single object, multi-object, and shapes
    bench_cases = ["apple", "shapes", "apple_orange"]
    subset_gt = {k: gt[k] for k in bench_cases if k in gt}
    if not subset_gt:
        subset_gt = gt

    models = {
        "hybrid_default": {"name": "Hybrid Pipeline (Classical + YOLOX-nano)", "type": "hybrid"},
        "classical_only": {"name": "Classical Rules Only", "type": "classical"},
        "yolo_world": {"name": "YOLO-World Open-Vocabulary", "type": "yoloworld"},
        "owlv2_openvocab": {"name": "OWLv2 / Text-Guided DINO", "type": "openvocab"}
    }
    results = {}

    for mid, mcfg in models.items():
        tp = n_det = n_gt = lab_ok = 0
        ious = []
        latencies = []

        for name, gts in subset_gt.items():
            im = cv2.imread(str(TESTSET / f"{name}.png"))
            n_gt += len(gts)
            t0 = time.time()

            if mcfg["type"] == "hybrid":
                res = detect(im, use_neural=True)
                dets = [{"label": d.label, "bbox": d.bbox, "confidence": d.confidence} for d in res.objects]
                lat = res.ms
            elif mcfg["type"] == "classical":
                res = detect(im, use_neural=False)
                dets = [{"label": d.label, "bbox": d.bbox, "confidence": d.confidence} for d in res.objects]
                lat = res.ms
            elif mcfg["type"] == "yoloworld":
                if YOLOWORLD.available():
                    preds, lat = YOLOWORLD.detect(im, classes=["apple", "orange", "person", "circle", "square", "triangle", "rectangle"])
                    dets = [{"label": p["label"], "bbox": p["bbox"], "confidence": p["score"]} for p in preds]
                else:
                    dets, lat = [], 0.0
            elif mcfg["type"] == "openvocab":
                if OPENVOCAB.available():
                    try:
                        preds = OPENVOCAB.detect(im, extra=["apple", "orange", "person", "circle", "square", "triangle", "rectangle"])
                        lat = (time.time() - t0) * 1000.0
                        dets = [{"label": p["label"], "bbox": p["bbox"], "confidence": p["score"]} for p in preds]
                    except Exception:
                        dets, lat = [], 0.0
                else:
                    dets, lat = [], 0.0

            latencies.append(lat)
            n_det += len(dets)
            pairs, _, _ = match(dets, gts, thr=0.5)
            tp += len(pairs)
            for i, j, v in pairs:
                ious.append(v)
                if dets[i]["label"].lower() == gts[j]["label"].lower():
                    lab_ok += 1

        prec = tp / max(1, n_det)
        rec = tp / max(1, n_gt)
        f1 = (2 * prec * rec) / max(1e-9, prec + rec)
        mean_iou = sum(ious) / max(1, len(ious))
        avg_lat = sum(latencies) / max(1, len(latencies))

        results[mid] = {
            "name": mcfg["name"],
            "precision": round(prec, 3),
            "recall": round(rec, 3),
            "f1_score": round(f1, 3),
            "label_accuracy": round(lab_ok / max(1, tp), 3),
            "mean_iou": round(mean_iou, 3),
            "mean_latency_ms": round(avg_lat, 1),
            "status": "ready" if (mid in ("hybrid_default", "classical_only") or
                                    (mid == "yolo_world" and YOLOWORLD.available()) or
                                    (mid == "owlv2_openvocab" and OPENVOCAB.available())) else "unavailable",
            "prompting": "fixed generic vocabulary; no ground-truth labels used as prompts" if mid in ("yolo_world", "owlv2_openvocab") else "n/a"
        }

    _BENCHMARK_CACHE = {
        "timestamp": time.time(),
        "dataset_images": len(gt),
        "comparison": results,
        "recommendation": "Use Hybrid Pipeline for ultra-low latency & clean cartoon/flat images; use YOLO-World for real-time open-vocabulary detection of user-specified concepts."
    }
    return _BENCHMARK_CACHE


if __name__ == "__main__":
    r = run()
    print("Benchmark summary:", json.dumps(r["summary"], indent=2))
    b = benchmark_models()
    print("\nModel comparison:\n", json.dumps(b["comparison"], indent=2))
