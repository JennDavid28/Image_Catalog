"""Colour naming from pixels. HSV rules (not nearest-palette) so shaded/lit variants of a hue share one name."""
from __future__ import annotations
import cv2
import numpy as np


def rgb_to_hsv_deg(rgb) -> tuple[float, float, float]:
    """rgb 0-255 -> (hue 0-360, sat 0-1, val 0-1)"""
    px = np.uint8([[list(map(int, np.clip(rgb, 0, 255)))]])
    h, s, v = cv2.cvtColor(px, cv2.COLOR_RGB2HSV)[0, 0]
    return float(h) * 2.0, float(s) / 255.0, float(v) / 255.0


def name_from_hsv(h: float, s: float, v: float) -> str:
    if v < 0.20:
        return "black"
    if s < 0.13:
        return "white" if v > 0.97 else ("light gray" if v > 0.86 else ("gray" if v > 0.35 else "black"))
    # chromatic
    if 10 <= h < 50 and s < 0.50 and v >= 0.62:
        return "peach"                       # light, desaturated orange: skin tones, cream
    if 10 <= h < 50 and v < 0.58 and s >= 0.35:
        return "brown"
    if h >= 290 or h < 10:                   # reds/magentas
        if s < 0.55 and v > 0.75:
            return "pink"
        return "purple" if 290 <= h < 330 else "red"
    if h < 26:
        return "red"
    if h < 46:
        return "orange"
    if h < 70:
        return "yellow"
    if h < 165:
        return "green"
    if h < 200:
        return "cyan"
    if h < 255:
        return "blue"
    return "purple"                           # 255-290


def name_rgb(rgb) -> str:
    return name_from_hsv(*rgb_to_hsv_deg(rgb))


def region_color(rgb_img: np.ndarray, mask: np.ndarray, erode: int = 0) -> tuple[str, tuple[int, int, int]]:
    """Median colour of mask pixels (optionally eroded to skip anti-aliased edges) -> (name, rgb)."""
    m = mask.astype(np.uint8)
    if erode:
        e = cv2.erode(m, np.ones((erode, erode), np.uint8))
        if e.sum() > 20:
            m = e
    px = rgb_img[m > 0]
    if len(px) == 0:
        return "unknown", (0, 0, 0)
    med = tuple(int(x) for x in np.median(px, axis=0))
    return name_rgb(med), med


def dominant_color_names(rgb_img: np.ndarray, mask: np.ndarray | None = None, k: int = 3) -> list[str]:
    """Top-k colour names by pixel share (used for image-level metadata)."""
    px = rgb_img[mask > 0] if mask is not None else rgb_img.reshape(-1, 3)
    if len(px) == 0:
        return []
    if len(px) > 6000:
        px = px[np.random.default_rng(0).choice(len(px), 6000, replace=False)]
    names: dict[str, int] = {}
    for c in cv2.cvtColor(px.reshape(-1, 1, 3).astype(np.uint8), cv2.COLOR_RGB2HSV).reshape(-1, 3):
        n = name_from_hsv(c[0] * 2.0, c[1] / 255.0, c[2] / 255.0)
        names[n] = names.get(n, 0) + 1
    tot = sum(names.values())
    return [n for n, c in sorted(names.items(), key=lambda kv: -kv[1]) if c / tot >= 0.04][:k]
