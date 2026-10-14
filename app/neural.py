"""Optional neural backend: YOLOX-nano (COCO, 80 classes) run through OpenCV's DNN module.

Why YOLOX-nano via cv2.dnn?  3.6 MB ONNX file, no torch/CUDA needed, ~30 ms on CPU.
It is trained on photographs, so it complements the classical detector (which is
strongest on flat / cartoon art with clean backgrounds).
"""
from __future__ import annotations
import os
from pathlib import Path
import cv2
import numpy as np

MODEL_PATH = Path(os.environ.get("YOLOX_ONNX", Path(__file__).resolve().parent.parent / "models" / "yolox_nano.onnx"))
INPUT = 416

COCO = ("person bicycle car motorcycle airplane bus train truck boat traffic_light fire_hydrant stop_sign "
        "parking_meter bench bird cat dog horse sheep cow elephant bear zebra giraffe backpack umbrella handbag "
        "tie suitcase frisbee skis snowboard sports_ball kite baseball_bat baseball_glove skateboard surfboard "
        "tennis_racket bottle wine_glass cup fork knife spoon bowl banana apple sandwich orange broccoli carrot "
        "hot_dog pizza donut cake chair couch potted_plant bed dining_table toilet tv laptop mouse remote keyboard "
        "cell_phone microwave oven toaster sink refrigerator book clock vase scissors teddy_bear hair_drier "
        "toothbrush").split()


def _grids():
    g, s = [], []
    for stride in (8, 16, 32):
        n = INPUT // stride
        xv, yv = np.meshgrid(np.arange(n), np.arange(n))
        grid = np.stack((xv, yv), 2).reshape(1, -1, 2)
        g.append(grid)
        s.append(np.full((1, grid.shape[1], 1), stride))
    return np.concatenate(g, 1), np.concatenate(s, 1)


class YoloxDetector:
    def __init__(self, path: Path = MODEL_PATH):
        self.path = Path(path)
        self._net = None
        self._grids, self._strides = _grids()

    def available(self) -> bool:
        return self.path.exists()

    def _load(self):
        if self._net is None:
            self._net = cv2.dnn.readNetFromONNX(str(self.path))
        return self._net

    def detect(self, bgr: np.ndarray, conf: float = 0.30, nms: float = 0.45):
        """Return [{'label','score','bbox':(x,y,w,h)}] in original-image pixels."""
        h, w = bgr.shape[:2]
        r = min(INPUT / h, INPUT / w)
        resized = cv2.resize(bgr, (int(w * r), int(h * r)), interpolation=cv2.INTER_LINEAR)
        canvas = np.full((INPUT, INPUT, 3), 114, np.uint8)       # YOLOX pads with 114, top-left aligned
        canvas[:resized.shape[0], :resized.shape[1]] = resized
        blob = canvas.astype(np.float32).transpose(2, 0, 1)[None]  # raw 0-255 BGR, CHW
        net = self._load()
        net.setInput(blob)
        out = net.forward()[0]                                    # (3549, 85) undecoded
        out = out.copy()
        out[:, :2] = (out[:, :2] + self._grids[0]) * self._strides[0]
        out[:, 2:4] = np.exp(out[:, 2:4]) * self._strides[0]
        scores_all = out[:, 4:5] * out[:, 5:]
        cls = scores_all.argmax(1)
        score = scores_all[np.arange(len(cls)), cls]
        keep = score > conf
        if not keep.any():
            return []
        cx, cy, bw, bh = out[keep, 0], out[keep, 1], out[keep, 2], out[keep, 3]
        boxes = np.stack([(cx - bw / 2) / r, (cy - bh / 2) / r, bw / r, bh / r], 1)
        sc, cl = score[keep], cls[keep]
        idx = cv2.dnn.NMSBoxes(boxes.tolist(), sc.tolist(), conf, nms)
        res = []
        for i in np.array(idx).flatten():
            x, y, bw_, bh_ = boxes[i]
            x0, y0 = max(0, x), max(0, y)
            x1, y1 = min(w, x + bw_), min(h, y + bh_)
            if x1 - x0 < 4 or y1 - y0 < 4:
                continue
            res.append({"label": COCO[int(cl[i])], "score": float(sc[i]),
                        "bbox": (int(x0), int(y0), int(x1 - x0), int(y1 - y0))})
        return sorted(res, key=lambda d: -d["score"])


def detect_tiled(yolox: "YoloxDetector", bgr: np.ndarray, conf: float = 0.30, nms: float = 0.45, min_side: int = 640):
    """Full-frame pass plus overlapping 2x2 tiles. Small objects in big photos (a cup on a desk) get ~2x more pixels
    per tile, which is what a 416 px nano model needs. Results are merged with class-wise NMS."""
    h, w = bgr.shape[:2]
    res = list(yolox.detect(bgr, conf=conf, nms=nms))
    if min(h, w) >= min_side:
        tw, th = int(w * 0.6), int(h * 0.6)
        for ox in (0, w - tw):
            for oy in (0, h - th):
                for r in yolox.detect(bgr[oy:oy + th, ox:ox + tw], conf=conf, nms=nms):
                    x, y, bw, bh = r["bbox"]
                    # drop boxes cut by an interior tile edge: they are partial views of something bigger
                    cut = (x <= 2 and ox > 0) or (y <= 2 and oy > 0) or (x + bw >= tw - 2 and ox == 0) or (y + bh >= th - 2 and oy == 0)
                    if not cut:
                        res.append({**r, "bbox": (x + ox, y + oy, bw, bh)})
    if len(res) <= 1:
        return res
    out = []
    for lab in {r["label"] for r in res}:
        grp = [r for r in res if r["label"] == lab]
        idx = cv2.dnn.NMSBoxes([list(r["bbox"]) for r in grp], [r["score"] for r in grp], conf, nms)
        out += [grp[i] for i in np.array(idx).flatten()]
    return sorted(out, key=lambda d: -d["score"])
