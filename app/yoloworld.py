"""YOLO-World Open-Vocabulary Object Detector & Benchmark Integration.

Provides real-time open-vocabulary object detection using Ultralytics YOLO-World.
Supports:
- User-specified custom categories & phrases ("red apple", "brand logo", "girl wearing a hat")
- Dynamic class prompting (zero-shot transfer without retraining)
- User-adjustable confidence thresholds
- NMS filtering & latency tracking
"""
from __future__ import annotations
import os
import threading
import time
from typing import List, Optional, Tuple, Dict, Any
import cv2
import numpy as np

YOLOWORLD_MODEL_NAME = os.environ.get("YOLOWORLD_MODEL", "yolov8s-worldv2.pt")


class YoloWorldDetector:
    def __init__(self, model_name: str = YOLOWORLD_MODEL_NAME):
        self.model_name = model_name
        self._model = None
        self._classes = None
        self._lock = threading.Lock()
        self._available = None
        self._error_msg = None

    def status(self) -> dict:
        if os.environ.get("YOLOWORLD", "").lower() == "off":
            return {"available": False, "reason": "disabled (YOLOWORLD=off)"}
        try:
            from ultralytics import YOLO
            return {"available": True, "reason": f"model: {self.model_name}"}
        except ImportError:
            return {"available": False, "reason": "ultralytics package not installed"}

    def available(self) -> bool:
        return self.status()["available"]

    def _load(self):
        with self._lock:
            if self._model is None:
                try:
                    from ultralytics import YOLO
                    # Load YOLO-World model
                    self._model = YOLO(self.model_name)
                    self._available = True
                except Exception as e:
                    self._available = False
                    self._error_msg = str(e)
                    raise
        return self._model

    def detect(
        self,
        bgr: np.ndarray,
        classes: Optional[List[str]] = None,
        conf_threshold: float = 0.25,
        iou_threshold: float = 0.5
    ) -> Tuple[List[Dict[str, Any]], float]:
        """Detect objects for arbitrary text queries.
        Returns (list_of_detections, inference_time_ms).
        Each detection: {'label', 'score', 'bbox': (x, y, w, h), 'src': 'yolo-world'}.
        """
        if not self.available():
            return [], 0.0

        t0 = time.time()
        try:
            model = self._load()
            # If custom classes are provided, set them dynamically
            if classes:
                clean_classes = [c.strip() for c in classes if c and c.strip()]
                if clean_classes and clean_classes != self._classes:
                    model.set_classes(clean_classes)
                    self._classes = clean_classes

            # Run inference
            results = model.predict(
                source=bgr,
                conf=conf_threshold,
                iou=iou_threshold,
                verbose=False
            )
            elapsed_ms = (time.time() - t0) * 1000.0

            detections = []
            if results and len(results) > 0:
                r = results[0]
                boxes = r.boxes.xyxy.cpu().numpy()
                scores = r.boxes.conf.cpu().numpy()
                class_ids = r.boxes.cls.cpu().numpy().astype(int)
                names = r.names

                for box, score, cid in zip(boxes, scores, class_ids):
                    x0, y0, x1, y1 = box
                    x, y, w, h = int(x0), int(y0), int(x1 - x0), int(y1 - y0)
                    label = names.get(cid, f"object_{cid}") if isinstance(names, dict) else names[cid]
                    detections.append({
                        "label": str(label),
                        "score": float(score),
                        "bbox": (x, y, w, h),
                        "src": "yolo-world"
                    })
            return detections, elapsed_ms
        except Exception:
            # Fall back to empty result on download/device exceptions
            return [], (time.time() - t0) * 1000.0


# Global singleton instance
YOLOWORLD = YoloWorldDetector()
