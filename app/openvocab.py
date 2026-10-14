"""Optional open-vocabulary detector (OWLv2 through Hugging Face transformers).

Why this exists: YOLOX-nano only knows the 80 COCO classes, so it can never say "pencil", "paper", "cable" or
"guitar". An open-vocabulary model scores an image against ANY list of words, which is the only practical route to
"label whatever is in the picture". It is optional because it needs torch + transformers and ~600 MB of weights.

Enable:   pip install torch transformers          (then first use downloads google/owlv2-base-patch16-ensemble)
Offline:  set OPENVOCAB_MODEL=/path/to/local/owlv2 directory
Disable:  set OPENVOCAB=off

The vocabulary is a long default list of everyday nouns plus anything the user types in "Also look for".
Detections go through the same GrabCut mask, colour naming, confidence and review-flag path as every other object.
"""
from __future__ import annotations
import importlib.util
import os
import threading

import numpy as np

MODEL_ID = os.environ.get("OPENVOCAB_MODEL", "google/owlv2-base-patch16-ensemble")

DEFAULT_VOCAB = """person man woman child baby face hand dog cat bird horse cow sheep elephant bear zebra giraffe fish insect
butterfly car truck bus van bicycle motorcycle scooter airplane helicopter train boat ship traffic light stop sign
bench chair sofa couch bed table desk shelf cabinet drawer lamp mirror clock vase potted plant flower tree bush
laptop computer monitor keyboard computer mouse tablet phone smartphone remote control camera television speaker
headphones microphone printer router cable charger plug battery light bulb
cup mug glass bottle can jar bowl plate fork knife spoon pan pot kettle teapot toaster microwave oven refrigerator sink
apple banana orange lemon strawberry grapes pineapple watermelon tomato carrot broccoli potato onion bread cake pizza
sandwich burger donut cookie egg cheese
book notebook paper crumpled paper sticky note pencil pen marker eraser ruler scissors tape stapler envelope folder
bag backpack handbag suitcase wallet umbrella hat cap helmet shoe boot sandal sock shirt jacket pants dress glasses
sunglasses watch ring necklace belt tie
ball football basketball tennis racket bat skateboard surfboard ski guitar piano violin drum toy teddy bear doll
toothbrush towel soap pillow blanket curtain rug door window stairs fence
key coin card box bucket broom tool hammer screwdriver wrench ladder shovel
sign flag poster painting statue
""".replace("\n", " ")


def _vocab_list(extra=None) -> list[str]:
    # the block above is whitespace-separated, so known two-word names are re-joined into phrases
    phrases = ["crumpled paper", "sticky note", "traffic light", "stop sign", "computer mouse", "remote control",
               "light bulb", "tennis racket", "teddy bear", "potted plant"]
    toks = DEFAULT_VOCAB.split()
    out, i = [], 0
    while i < len(toks):
        two = " ".join(toks[i:i + 2])
        if two in phrases:
            out.append(two); i += 2
        else:
            out.append(toks[i]); i += 1
    seen, res = set(), []
    for w in out + [e.strip().lower() for e in (extra or []) if e and e.strip()]:
        if w not in seen:
            seen.add(w); res.append(w)
    return res


class OpenVocabDetector:
    def __init__(self):
        self._model = self._proc = None
        self._failed = None
        self._lock = threading.Lock()

    def status(self) -> dict:
        if os.environ.get("OPENVOCAB", "").lower() == "off":
            return {"available": False, "reason": "disabled (OPENVOCAB=off)"}
        if self._failed:
            return {"available": False, "reason": self._failed}
        if not (importlib.util.find_spec("torch") and importlib.util.find_spec("transformers")):
            return {"available": False, "reason": "torch and transformers are not installed"}
        return {"available": True, "reason": f"model: {MODEL_ID}"}

    def available(self) -> bool:
        return self.status()["available"]

    def _load(self):
        with self._lock:
            if self._model is None:
                try:
                    from transformers import Owlv2ForObjectDetection, Owlv2Processor
                    self._proc = Owlv2Processor.from_pretrained(MODEL_ID)
                    self._model = Owlv2ForObjectDetection.from_pretrained(MODEL_ID).eval()
                except Exception as e:                      # offline, missing weights, out of memory ...
                    self._failed = f"could not load {MODEL_ID}: {type(e).__name__}"
                    raise
        return self._model, self._proc

    def detect(self, bgr: np.ndarray, extra=None, threshold: float = 0.22, nms: float = 0.5):
        """Return [{'label','score','bbox':(x,y,w,h)}] in pixels of `bgr`. Empty list if the backend is unusable."""
        if not self.available():
            return []
        try:
            import cv2
            import torch
            from PIL import Image
            model, proc = self._load()
            queries = _vocab_list(extra)
            h, w = bgr.shape[:2]
            pil = Image.fromarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            inputs = proc(text=[[f"a photo of a {q}" for q in queries]], images=pil, return_tensors="pt")
            with torch.no_grad():
                out = model(**inputs)
            side = max(h, w)                                # OWLv2 pads to a square, boxes are relative to it
            res = proc.post_process_object_detection(out, threshold=threshold, target_sizes=torch.tensor([[side, side]]))[0]
        except Exception:
            return []
        boxes, scores, labels = res["boxes"].tolist(), res["scores"].tolist(), res["labels"].tolist()
        items = []
        for b, s, l in zip(boxes, scores, labels):
            x0, y0, x1, y1 = max(0, b[0]), max(0, b[1]), min(w, b[2]), min(h, b[3])
            if x1 - x0 >= 6 and y1 - y0 >= 6:
                items.append({"label": queries[l], "score": float(s), "bbox": (int(x0), int(y0), int(x1 - x0), int(y1 - y0))})
        return nms_items(items, nms)


def nms_items(items: list, thr: float = 0.5) -> list:
    """Greedy NMS across all labels (one box per physical object)."""
    items = sorted(items, key=lambda d: -d["score"])
    keep = []
    for it in items:
        if all(_iou(it["bbox"], k["bbox"]) < thr for k in keep):
            keep.append(it)
    return keep


def _iou(a, b) -> float:
    ax, ay, aw, ah = a; bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx)); iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / float(aw * ah + bw * bh - inter + 1e-9)


OPENVOCAB = OpenVocabDetector()
