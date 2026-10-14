"""Mask extraction and foreground segmentation (rembg U²-Net, GrabCut, Lab-matte)."""
from __future__ import annotations
import cv2
import numpy as np


def to_lab(bgr: np.ndarray) -> np.ndarray:
    """float32 Lab (L 0-100, a/b ~ -127..127)"""
    return cv2.cvtColor(bgr.astype(np.float32) / 255.0, cv2.COLOR_BGR2LAB)


def border_background(lab: np.ndarray):
    """median border colour + 90th-percentile deviation (how 'flat' the background is)."""
    h, w = lab.shape[:2]
    b = max(2, int(0.01 * min(h, w)))
    border = np.concatenate([lab[:b].reshape(-1, 3), lab[-b:].reshape(-1, 3),
                             lab[:, :b].reshape(-1, 3), lab[:, -b:].reshape(-1, 3)])
    bg = np.median(border, axis=0)
    spread = float(np.percentile(np.linalg.norm(border - bg, axis=1), 90))
    return bg, spread


def fill_holes(mask: np.ndarray) -> np.ndarray:
    """Fill interior holes: flood the background from outside a 1px pad, everything unreached is object."""
    m = (mask > 0).astype(np.uint8)
    h, w = m.shape
    ff = cv2.copyMakeBorder(m, 1, 1, 1, 1, cv2.BORDER_CONSTANT, value=0)
    cv2.floodFill(ff, np.zeros((h + 4, w + 4), np.uint8), (0, 0), 2)
    return ff[1:-1, 1:-1] != 2


def largest_component(mask: np.ndarray) -> np.ndarray:
    n, lab, stats, _ = cv2.connectedComponentsWithStats(mask.astype(np.uint8), connectivity=8)
    if n <= 1:
        return mask.astype(bool)
    k = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return lab == k


def rembg_cutout(bgr: np.ndarray) -> np.ndarray:
    """Extract foreground using rembg (U²-Net). Returns BGRA uint8 array."""
    import pathlib
    import os
    u2net_file = pathlib.Path.home() / ".u2net" / "u2net.onnx"
    custom_path = os.environ.get("U2NET_PATH")
    if u2net_file.exists() or (custom_path and os.path.exists(custom_path)):
        try:
            import rembg
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            result = rembg.remove(rgb)
            return cv2.cvtColor(result, cv2.COLOR_RGBA2BGRA)
        except Exception:
            pass

    # High-quality GrabCut / Lab-matte fallback
    H, W = bgr.shape[:2]
    lab = to_lab(bgr)
    bg, spread = border_background(lab)
    if spread < 10:
        d = np.linalg.norm(lab - bg, axis=2)
        alpha = np.clip((d - 10) * 15, 0, 255).astype(np.uint8)
    else:
        gm, _ = grabcut_mask(bgr, (2, 2, W - 4, H - 4))
        alpha = (gm * 255).astype(np.uint8)
    return np.dstack([bgr, alpha])


def rembg_mask(bgr: np.ndarray) -> np.ndarray:
    """Extract boolean foreground mask using rembg."""
    bgra = rembg_cutout(bgr)
    return bgra[..., 3] > 64


def grabcut_mask(bgr: np.ndarray, bbox, iters: int = 4, max_side: int = 384):
    """GrabCut initialised with the box. Returns (bool mask in full-res, quality = mask area / box area)."""
    H, W = bgr.shape[:2]
    x, y, w, h = [int(v) for v in bbox]
    x, y = max(0, x), max(0, y)
    w, h = min(w, W - x), min(h, H - y)
    s = min(1.0, max_side / max(H, W))
    small = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else bgr.copy()
    pad = 12
    small = cv2.copyMakeBorder(small, pad, pad, pad, pad, cv2.BORDER_REPLICATE)
    rect = (int(x * s) + pad, int(y * s) + pad, max(4, int(w * s)), max(4, int(h * s)))
    mask = np.zeros(small.shape[:2], np.uint8)
    bgd, fgd = np.zeros((1, 65)), np.zeros((1, 65))
    try:
        cv2.setRNGSeed(0)
        cv2.grabCut(small, mask, rect, bgd, fgd, iters, cv2.GC_INIT_WITH_RECT)
    except cv2.error:
        return np.zeros((H, W), bool), 0.0
    fg = ((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD)).astype(np.uint8)[pad:-pad, pad:-pad]
    fg = cv2.resize(fg, (W, H), interpolation=cv2.INTER_NEAREST) if s < 1 else fg
    fg = fill_holes(largest_component(fg))
    return fg, float(fg.sum()) / max(1, w * h)


def box_mask(shape, bbox) -> np.ndarray:
    m = np.zeros(shape[:2], bool)
    x, y, w, h = [int(v) for v in bbox]
    m[max(0, y):y + h, max(0, x):x + w] = True
    return m


def segment_box(bgr: np.ndarray, bbox, prefer_rembg: bool = False):
    """Best-effort mask for a user-supplied/edited box. Chain: rembg -> background matte -> GrabCut -> plain box.
    Returns (mask, source) with source in {'rembg','matte','grabcut','box'}."""
    H, W = bgr.shape[:2]
    x, y, w, h = [int(round(v)) for v in bbox]
    x, y = max(0, x), max(0, y)
    w, h = max(2, min(w, W - x)), max(2, min(h, H - y))

    if prefer_rembg:
        sub = bgr[y:y + h, x:x + w]
        rm = rembg_mask(sub)
        if 0.10 <= rm.sum() / (w * h) <= 0.98:
            full = np.zeros((H, W), bool)
            full[y:y + h, x:x + w] = rm
            return full, "rembg"

    lab = to_lab(bgr)
    bg, spread = border_background(lab)
    if spread < 8:  # flat background: colour-distance matte, threshold adapted to the object's contrast
        d = np.linalg.norm(cv2.GaussianBlur(lab, (0, 0), 1.0) - bg, axis=2)
        sub = d[y:y + h, x:x + w]
        thr = max(4.0, min(14.0, 0.5 * float(np.percentile(sub, 95))))
        m = np.zeros((H, W), np.uint8)
        m[y:y + h, x:x + w] = (sub > thr)
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        m = fill_holes(m)
        if m.sum() > 0:
            m = largest_component(m) if _single_blob(m) else m
        frac = m.sum() / (w * h)
        if 0.10 <= frac <= 0.97:
            return m, "matte"
    gm, q = grabcut_mask(bgr, (x, y, w, h))
    if 0.20 <= q <= 0.97:
        return gm, "grabcut"
    return box_mask(bgr.shape, (x, y, w, h)), "box"


def _single_blob(m) -> bool:
    n, _, stats, _ = cv2.connectedComponentsWithStats(m.astype(np.uint8), connectivity=8)
    if n <= 2:
        return True
    areas = sorted(stats[1:, cv2.CC_STAT_AREA], reverse=True)
    return areas[0] > 5 * areas[1]


def mask_bbox(mask: np.ndarray):
    ys, xs = np.where(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)
