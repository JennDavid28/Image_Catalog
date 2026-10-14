"""Pixel-level editing operations. Pure functions on numpy arrays, no storage or HTTP here.

Every function takes and returns BGR (or BGRA when transparency is involved) uint8 arrays.
The colour-adjust maths mirrors the CSS filter definitions (hue-rotate, saturate, brightness, contrast), so the
live preview in the browser (which uses CSS filters) matches what the server writes to disk.
"""
from __future__ import annotations
import colorsys
import math

import cv2
import numpy as np

MAX_SIDE = 6000


class EditError(ValueError):
    pass


def _int(v, name):
    try:
        return int(round(float(v)))
    except Exception:
        raise EditError(f"{name} must be a number")


# ---------------------------------------------------------------------- geometry
def crop(img: np.ndarray, x, y, w, h) -> np.ndarray:
    H, W = img.shape[:2]
    x, y, w, h = _int(x, "x"), _int(y, "y"), _int(w, "width"), _int(h, "height")
    x, y = max(0, min(x, W - 1)), max(0, min(y, H - 1))
    w, h = min(w, W - x), min(h, H - y)
    if w < 4 or h < 4:
        raise EditError("crop area is too small (min 4 x 4 px)")
    return img[y:y + h, x:x + w].copy()


def resize(img: np.ndarray, width=None, height=None, percent=None) -> np.ndarray:
    H, W = img.shape[:2]
    if percent is not None:
        p = float(percent)
        if not 1 <= p <= 800:
            raise EditError("percent must be between 1 and 800")
        nw, nh = round(W * p / 100), round(H * p / 100)
    elif width and height:
        nw, nh = _int(width, "width"), _int(height, "height")
    elif width:
        nw = _int(width, "width"); nh = round(H * nw / W)
    elif height:
        nh = _int(height, "height"); nw = round(W * nh / H)
    else:
        raise EditError("give width, height or percent")
    if nw < 4 or nh < 4:
        raise EditError("target size is too small")
    if max(nw, nh) > MAX_SIDE:
        raise EditError(f"target size is too large (max {MAX_SIDE}px on a side)")
    interp = cv2.INTER_AREA if nw * nh < W * H else cv2.INTER_CUBIC
    return cv2.resize(img, (nw, nh), interpolation=interp)


def rotate(img: np.ndarray, degrees) -> np.ndarray:
    d = _int(degrees, "degrees") % 360
    if d == 0:
        return img.copy()
    if d == 90:
        return cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
    if d == 180:
        return cv2.rotate(img, cv2.ROTATE_180)
    if d == 270:
        return cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
    raise EditError("rotation must be 90, 180 or 270 degrees")


def flip(img: np.ndarray, axis: str) -> np.ndarray:
    if axis == "horizontal":
        return cv2.flip(img, 1)
    if axis == "vertical":
        return cv2.flip(img, 0)
    raise EditError("axis must be 'horizontal' or 'vertical'")


# ---------------------------------------------------------------------- colour
def _hue_matrix(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([
        [0.213 + c * 0.787 - s * 0.213, 0.715 - c * 0.715 - s * 0.715, 0.072 - c * 0.072 + s * 0.928],
        [0.213 - c * 0.213 + s * 0.143, 0.715 + c * 0.285 + s * 0.140, 0.072 - c * 0.072 - s * 0.283],
        [0.213 - c * 0.213 - s * 0.787, 0.715 - c * 0.715 + s * 0.715, 0.072 + c * 0.928 + s * 0.072]], np.float32)


def _sat_matrix(s: float) -> np.ndarray:
    return np.array([
        [0.213 + 0.787 * s, 0.715 - 0.715 * s, 0.072 - 0.072 * s],
        [0.213 - 0.213 * s, 0.715 + 0.285 * s, 0.072 - 0.072 * s],
        [0.213 - 0.213 * s, 0.715 - 0.715 * s, 0.072 + 0.928 * s]], np.float32)


def adjust(img: np.ndarray, hue=0.0, saturation=100.0, brightness=100.0, contrast=100.0, grayscale=False,
           sepia=0.0, invert=False) -> np.ndarray:
    """CSS-compatible adjustments. hue in degrees (-180..180), the rest in percent (100 = unchanged)."""
    hue, sat, bri, con = float(hue), float(saturation) / 100, float(brightness) / 100, float(contrast) / 100
    if not (-180 <= hue <= 180 and 0 <= sat <= 5 and 0 <= bri <= 3 and 0 <= con <= 3):
        raise EditError("adjustment value out of range")
    alpha = img[..., 3:] if img.shape[2] == 4 else None
    rgb = cv2.cvtColor(img[..., :3], cv2.COLOR_BGR2RGB).astype(np.float32) / 255
    if grayscale:
        rgb = np.repeat((rgb @ np.array([0.2126, 0.7152, 0.0722], np.float32))[..., None], 3, axis=2)
    if hue:
        rgb = rgb @ _hue_matrix(hue).T
    if sat != 1:
        rgb = rgb @ _sat_matrix(sat).T
    if sepia:
        k = min(1.0, float(sepia) / 100)
        m = np.array([[0.393, 0.769, 0.189], [0.349, 0.686, 0.168], [0.272, 0.534, 0.131]], np.float32)
        rgb = rgb * (1 - k) + (rgb @ m.T) * k
    rgb = np.clip(rgb, 0, 1)
    if bri != 1:
        rgb = rgb * bri
    if con != 1:
        rgb = (rgb - 0.5) * con + 0.5
    if invert:
        rgb = 1 - np.clip(rgb, 0, 1)
    out = cv2.cvtColor((np.clip(rgb, 0, 1) * 255 + 0.5).astype(np.uint8), cv2.COLOR_RGB2BGR)
    return np.dstack([out, alpha]) if alpha is not None else out


def hex_to_rgb(h: str) -> tuple:
    h = (h or "").strip().lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    if len(h) != 6 or any(c not in "0123456789abcdefABCDEF" for c in h):
        raise EditError("colour must be a hex value like #ff8800")
    return tuple(int(h[i:i + 2], 16) for i in (0, 2, 4))


def recolor_region(img: np.ndarray, mask: np.ndarray, hex_color: str, strength: float = 1.0) -> np.ndarray:
    """Re-tint only the masked pixels to a target colour while keeping their shading.

    Hue and saturation come from the target; brightness (V) is kept, so highlights and shadows survive.
    Targets with no saturation (white, grey, black) are handled by scaling brightness instead of hue.
    The mask edge is feathered so the recoloured area blends in.
    """
    r, g, b = hex_to_rgb(hex_color)
    th, ts, tv = colorsys.rgb_to_hsv(r / 255, g / 255, b / 255)
    bgr = img[..., :3]
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV).astype(np.float32)
    if ts < 0.08:                                       # neutral target (white / grey / black): drop colour, shift brightness
        hsv[..., 1] = 0
        v = hsv[..., 2]
        if tv >= 0.85:
            hsv[..., 2] = 255 - (255 - v) * 0.45
        elif tv <= 0.25:
            hsv[..., 2] = v * 0.30
        else:
            hsv[..., 2] = np.clip(v * (tv / max(0.05, float(v[mask > 0].mean()) / 255 if mask.any() else 0.5)), 0, 255)
    else:
        hsv[..., 0] = th * 179
        hsv[..., 1] = np.clip(np.maximum(hsv[..., 1] * 0.35, ts * 255) * 1.0, 0, 255)
    out = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2BGR)
    soft = cv2.GaussianBlur((mask > 0).astype(np.float32), (0, 0), 1.2)[..., None] * float(strength)
    res = (out.astype(np.float32) * soft + bgr.astype(np.float32) * (1 - soft)).astype(np.uint8)
    return np.dstack([res, img[..., 3]]) if img.shape[2] == 4 else res


# ---------------------------------------------------------------------- cut / erase
def cutout(img: np.ndarray, mask: np.ndarray, pad: int = 0) -> np.ndarray:
    """BGRA image containing only the masked pixels (transparent elsewhere), cropped to the mask."""
    m = (mask > 0).astype(np.uint8) * 255
    if m.sum() == 0:
        raise EditError("nothing to cut: the object mask is empty")
    a = cv2.GaussianBlur(cv2.erode(m, np.ones((3, 3), np.uint8)), (0, 0), 0.8)
    ys, xs = np.where(a > 8)
    H, W = a.shape
    y0, y1, x0, x1 = max(0, ys.min() - pad), min(H, ys.max() + 1 + pad), max(0, xs.min() - pad), min(W, xs.max() + 1 + pad)
    bgr = img[..., :3]
    return np.dstack([bgr, a])[y0:y1, x0:x1].copy()


def erase(img: np.ndarray, mask: np.ndarray, radius: int = 5) -> np.ndarray:
    """Remove the masked pixels and fill them from the surroundings (Telea inpainting)."""
    m = cv2.dilate((mask > 0).astype(np.uint8) * 255, np.ones((7, 7), np.uint8))
    if m.sum() == 0:
        raise EditError("nothing to erase")
    if m.mean() / 255 > 0.6:
        raise EditError("the selection covers most of the image; erasing it would leave nothing to fill from")
    bgr = cv2.inpaint(img[..., :3], m, radius, cv2.INPAINT_TELEA)
    return np.dstack([bgr, img[..., 3]]) if img.shape[2] == 4 else bgr


def remove_background(img: np.ndarray, masks: list[np.ndarray]) -> np.ndarray:
    """Keep every listed object, make everything else transparent."""
    if not masks:
        raise EditError("no objects to keep")
    union = np.logical_or.reduce([m > 0 for m in masks])
    m = union.astype(np.uint8) * 255
    a = cv2.GaussianBlur(cv2.erode(m, np.ones((3, 3), np.uint8)), (0, 0), 0.8)
    return np.dstack([img[..., :3], a])


def blur_background(img: np.ndarray, masks: list[np.ndarray], ksize: int = 25) -> np.ndarray:
    """Blur the background (bokeh effect) while preserving crisp foreground objects."""
    if not masks:
        raise EditError("no objects to keep in focus")
    union = np.logical_or.reduce([m > 0 for m in masks])
    k = max(3, ksize if ksize % 2 == 1 else ksize + 1)
    blurred_bg = cv2.GaussianBlur(img[..., :3], (k, k), 0)
    # Soft alpha blend for foreground edge feathering
    m_soft = cv2.GaussianBlur(union.astype(np.float32), (5, 5), 1.5)[..., None]
    fg = img[..., :3].astype(np.float32)
    bg = blurred_bg.astype(np.float32)
    blended = np.clip(fg * m_soft + bg * (1.0 - m_soft), 0, 255).astype(np.uint8)
    return np.dstack([blended, img[..., 3]]) if img.shape[2] == 4 else blended


def replace_background(img: np.ndarray, masks: list[np.ndarray], bg: np.ndarray | tuple | str) -> np.ndarray:
    """Replace background pixels with a solid color, gradient, or another image."""
    if not masks:
        raise EditError("no foreground objects specified")
    H, W = img.shape[:2]
    union = np.logical_or.reduce([m > 0 for m in masks])
    m_soft = cv2.GaussianBlur(union.astype(np.float32), (3, 3), 1.0)[..., None]

    if isinstance(bg, np.ndarray):
        # Resize background image to match current canvas
        bg_canvas = cv2.resize(bg[..., :3], (W, H), interpolation=cv2.INTER_AREA)
    elif isinstance(bg, (tuple, list)):
        bg_canvas = np.full((H, W, 3), bg[:3], dtype=np.uint8)
    elif isinstance(bg, str) and bg.startswith("#"):
        hex_str = bg.lstrip("#")
        rgb = tuple(int(hex_str[i:i + 2], 16) for i in (0, 2, 4))
        bgr = (rgb[2], rgb[1], rgb[0])
        bg_canvas = np.full((H, W, 3), bgr, dtype=np.uint8)
    else:
        bg_canvas = np.full((H, W, 3), (255, 255, 255), dtype=np.uint8)

    fg = img[..., :3].astype(np.float32)
    new_bg = bg_canvas.astype(np.float32)
    blended = np.clip(fg * m_soft + new_bg * (1.0 - m_soft), 0, 255).astype(np.uint8)
    return blended


def rembg_auto_remove_bg(img: np.ndarray) -> np.ndarray:
    """Automatic AI background removal via rembg (U²-Net)."""
    from .segmentation import rembg_cutout
    return rembg_cutout(img[..., :3])


def encode(img: np.ndarray, fmt: str = "png") -> bytes:
    if img.shape[2] == 4 and fmt != "png":
        a = img[..., 3:4].astype(np.float32) / 255
        img = (img[..., :3] * a + 255 * (1 - a)).astype(np.uint8)
    ext = {"png": ".png", "jpg": ".jpg", "jpeg": ".jpg", "webp": ".webp"}.get(fmt)
    if not ext:
        raise EditError("format must be png, jpg or webp")
    ok, buf = cv2.imencode(ext, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise EditError("could not encode the result")
    return buf.tobytes()
