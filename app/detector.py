"""Hybrid object detector.

Router
  flat background  -> classical colour/shape pipeline (strong on cartoons, flat art, product shots)
                      (+ YOLOX corroboration when the neural model is available)
  complex backdrop -> YOLOX-nano proposals + GrabCut masks (photographs)
  nothing found on a flat background -> YOLOX as a second opinion

Every detection carries `evidence` (why this label/colour) and a confidence = label certainty x evidence quality,
so low-certainty guesses (e.g. boy/girl from a silhouette) are visibly flagged instead of pretending to be sure.
"""
from __future__ import annotations
import time
from dataclasses import dataclass, field
import cv2
import numpy as np

from .colors import dominant_color_names, name_from_hsv, region_color, rgb_to_hsv_deg
from .neural import YoloxDetector, detect_tiled
from .openvocab import OPENVOCAB, nms_items
from .segmentation import border_background, box_mask, fill_holes, grabcut_mask, mask_bbox, to_lab

DETECTOR_VERSION = "hybrid-2.0 (classical + YOLOX-nano tiled + optional open-vocabulary + GrabCut)"
NEEDS_REVIEW = 0.60          # below this the UI flags a detection for human review
FLAT_BG_SPREAD = 8.0         # border deviation (Lab dE, p90) under which the background counts as flat
FLAT_FILL_STD = 2.8          # strongest Lab-channel std inside an object under which its fill counts as flat
FG_THRESHOLD = 14.0          # Lab dE from background that counts as foreground
ROUND_CLASSES = {"apple", "orange", "sports ball", "donut", "clock", "bowl", "vase"}


@dataclass(eq=False)
class Detection:
    label: str
    color: str
    confidence: float
    bbox: tuple                       # (x, y, w, h)
    mask: np.ndarray                  # bool, image-sized
    source: str                       # classical | neural
    attrs: dict = field(default_factory=dict)
    evidence: list = field(default_factory=list)


@dataclass
class DetectionResult:
    status: str                       # ok | low_confidence | no_objects | failed
    objects: list
    route: str
    message: str
    ms: float
    dominant_colors: list


@dataclass(eq=False)
class Region:
    mask: np.ndarray
    area: int
    rgb: tuple
    color: str
    cx: float
    cy: float
    bbox: tuple
    parts: list = field(default_factory=list)


# ----------------------------------------------------------------------------- small helpers
def iou(a, b) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / float(aw * ah + bw * bh - inter + 1e-9)


def _lab_to_rgb(c) -> tuple:
    return tuple(float(v) * 255 for v in cv2.cvtColor(np.float32([[c]]), cv2.COLOR_LAB2RGB)[0, 0])


def _is_skin(rgb) -> bool:
    h, s, v = rgb_to_hsv_deg(rgb)
    return 10 <= h <= 45 and 0.15 <= s <= 0.65 and v >= 0.45


def _hue_dist(a, b) -> float:
    d = abs(a - b)
    return min(d, 360 - d)


def _contour_stats(mask: np.ndarray) -> dict:
    cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    c = max(cnts, key=cv2.contourArea)
    area, per = cv2.contourArea(c), cv2.arcLength(c, True)
    hull = cv2.convexHull(c)
    (_, _), (rw, rh), _ = cv2.minAreaRect(c)
    hull_area, hull_per = cv2.contourArea(hull), cv2.arcLength(hull, True)
    return dict(c=c, area=area, per=per, hull=hull, circ=4 * np.pi * area / (per * per + 1e-9),
                hull_circ=4 * np.pi * hull_area / (hull_per * hull_per + 1e-9),
                solidity=area / (cv2.contourArea(hull) + 1e-9), rect_fill=area / (rw * rh + 1e-9),
                aspect=min(rw, rh) / (max(rw, rh) + 1e-9))


def _fit_circle(pts: np.ndarray):
    """Least-squares (Kasa) circle fit. Returns (cx, cy, r, relative residual, angular extent in rad)."""
    x, y = pts[:, 0].astype(float), pts[:, 1].astype(float)
    sol, *_ = np.linalg.lstsq(np.c_[x, y, np.ones_like(x)], -(x ** 2 + y ** 2), rcond=None)
    cx, cy = -sol[0] / 2, -sol[1] / 2
    r = float(np.sqrt(max(cx ** 2 + cy ** 2 - sol[2], 1e-9)))
    res = float(np.std(np.hypot(x - cx, y - cy) - r) / r)
    ang = np.sort(np.arctan2(y - cy, x - cx))
    gaps = np.diff(np.r_[ang, ang[0] + 2 * np.pi])
    return cx, cy, r, res, float(2 * np.pi - gaps.max())


def _line_intersection(p1, p2, p3, p4):
    d1, d2 = np.subtract(p2, p1).astype(float), np.subtract(p4, p3).astype(float)
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-6:
        return None
    t = ((p3[0] - p1[0]) * d2[1] - (p3[1] - p1[1]) * d2[0]) / den
    return np.add(p1, t * d1)


def _parallel(a, b, tol=10) -> bool:
    ang = abs(np.degrees(np.arctan2(a[0] * b[1] - a[1] * b[0], float(np.dot(a, b)))))
    return ang < tol or ang > 180 - tol


# ----------------------------------------------------------------------------- colour regions
def _color_regions(rgb: np.ndarray, lab: np.ndarray, comp: np.ndarray) -> list[Region]:
    """Split one connected foreground component into flat-ish colour regions.
    k-means in Lab, then merge clusters that are shading variants of one hue (same hue, similar saturation)."""
    ys, xs = np.where(comp)
    pix = lab[ys, xs]
    n = len(pix)
    sample = pix if n <= 5000 else pix[np.random.default_rng(0).choice(n, 5000, replace=False)]
    k = int(min(7, max(1, len(np.unique(sample.round(0), axis=0)))))
    cv2.setRNGSeed(0)
    _, lbl, centers = cv2.kmeans(sample.astype(np.float32), k, None,
                                 (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 0.5), 3, cv2.KMEANS_PP_CENTERS)
    counts = np.bincount(lbl.ravel(), minlength=k).astype(float)
    hsv = [rgb_to_hsv_deg(_lab_to_rgb(c)) for c in centers]
    parent = list(range(k))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    for i in range(k):
        for j in range(i + 1, k):
            de = float(np.linalg.norm(centers[i] - centers[j]))
            (hi, si, vi), (hj, sj, vj) = hsv[i], hsv[j]
            chroma = si > 0.30 and sj > 0.30 and vi > 0.2 and vj > 0.2
            if de < 14 or (chroma and _hue_dist(hi, hj) < 20 and abs(si - sj) < 0.25 and de < 45):
                parent[find(j)] = find(i)
    groups = {}
    for i in range(k):
        groups.setdefault(find(i), []).append(i)
    gc = np.array([np.average(centers[m], axis=0, weights=counts[m]) for m in groups.values()])
    gshare = np.array([counts[m].sum() for m in groups.values()]) / counts.sum()
    keep = gshare >= 0.0025
    gc = gc[keep] if keep.any() else gc
    d = np.linalg.norm(pix[:, None, :] - gc[None, :, :], axis=2)
    assign = d.argmin(1)
    lm = np.zeros(comp.shape, np.int32)
    lm[ys, xs] = assign + 1

    regions, min_area = [], max(25, int(0.0025 * n))
    for g in range(len(gc)):
        m = (lm == g + 1).astype(np.uint8)
        if m.sum() > 400:
            m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        cnt, cl, st, cen = cv2.connectedComponentsWithStats(m, connectivity=8)
        for i in range(1, cnt):
            if st[i, cv2.CC_STAT_AREA] < min_area:
                continue
            rm = cl == i
            name, rgbv = region_color(rgb, rm, erode=5)
            regions.append(Region(rm, int(st[i, cv2.CC_STAT_AREA]), rgbv, name, float(cen[i][0]), float(cen[i][1]),
                                  tuple(int(v) for v in st[i, :4])))
    return regions


def _partition(comp: np.ndarray, masks: list[np.ndarray]) -> list[np.ndarray]:
    """Give every component pixel to its nearest region (fills AA fringes / dropped specks)."""
    lm = np.zeros(comp.shape, np.int32)
    for i, m in enumerate(masks):
        lm[m] = i + 1
    seeds = lm > 0
    _, ids = cv2.distanceTransformWithLabels((~seeds).astype(np.uint8), cv2.DIST_L2, 3, labelType=cv2.DIST_LABEL_PIXEL)
    lookup = np.zeros(int(ids.max()) + 1, np.int32)
    lookup[ids[seeds]] = lm[seeds]
    full = lookup[ids]
    return [(full == i + 1) & comp for i in range(len(masks))]


# ----------------------------------------------------------------------------- figure (cartoon person) rule
def _try_figure(comp, regions, bb, dist):
    bx, by, bw, bh = bb
    if bh < 1.3 * bw:
        return None
    skins = [r for r in regions if _is_skin(r.rgb)]
    if not skins:
        return None
    head = min(skins, key=lambda r: r.cy)
    hx, hy, hw, hh = head.bbox
    if hy - by > 0.15 * bh:
        return None
    body = [r for r in regions if r is not head and not _is_skin(r.rgb) and r.cy > head.cy
            and r.bbox[0] < hx + hw and r.bbox[0] + r.bbox[2] > hx and r.area >= 0.2 * head.area]
    if not body:
        return None
    hair = [r for r in regions if not _is_skin(r.rgb) and r.cy < hy + 0.55 * hh
            and r.bbox[0] < hx + 1.25 * hw and r.bbox[0] + r.bbox[2] > hx - 0.25 * hw]
    rest = [r for r in regions if r not in hair and not _is_skin(r.rgb) and r.area > 0.02 * comp.sum()]
    frac = lambda r: (r.cy - by) / bh
    tops = [r for r in rest if 0.2 <= frac(r) <= 0.55]
    bots = [r for r in rest if 0.5 <= frac(r) <= 0.88]
    shoes = [r for r in regions if not _is_skin(r.rgb) and frac(r) > 0.88]
    top = max(tops, key=lambda r: r.area) if tops else None
    bot = max(bots, key=lambda r: r.area) if bots else None

    def widths(lo, hi):
        return [int(comp[int(by + t * (bh - 1))].sum()) for t in np.linspace(lo, hi, 12)]

    # widest row in the lower body (a skirt flares to its hem) vs typical torso width (arms included)
    flare = max(widths(0.64, 0.82)) / (float(np.median(widths(0.36, 0.46))) + 1e-9)
    label = "girl" if flare > 0.90 else "boy"
    clothing = {k: v.color for k, v in (("top", top), ("bottom", bot)) if v is not None}
    if shoes:
        clothing["shoes"] = max(shoes, key=lambda r: r.area).color
    colour = "/".join(clothing[k] for k in ("top", "bottom") if k in clothing) or "multicolor"
    evid = ["figure rule: round skin-tone head at top with clothing regions below -> cartoon person",
            f"silhouette flare ratio {flare:.2f} ({'skirt/dress' if label == 'girl' else 'trousers'}) -> '{label}' "
            "(cartoon-convention heuristic, low certainty - verify)"]
    c = float(np.clip(np.median(dist[comp]) / 40, 0, 1))
    conf = round(0.58 * (0.6 + 0.2 * c + 0.2 * min(1.0, 0.5 + 0.5 * (len(clothing) / 2))), 2)
    attrs = {"kind": "cartoon person", "clothing": clothing, "skin": head.color,
             "hair": max(hair, key=lambda r: r.area).color if hair else None, "silhouette_flare": round(flare, 2)}
    return Detection(label, colour, conf, tuple(bb), comp, "classical", attrs, evid)


# ----------------------------------------------------------------------------- shape / fruit classification
def _classify_region(rgb, mask, others, regions_parts, flat_std, dist) -> dict:
    st = _contour_stats(mask)
    core = cv2.erode(mask.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    core = core if core.sum() > 30 else mask
    color, rgbv = region_color(rgb, mask, erode=7)
    perim = max(st["per"], 1.0)
    contact = 0.0
    free_pts = None
    if others is not None and others.any():
        near = cv2.dilate(mask.astype(np.uint8), np.ones((3, 3), np.uint8), iterations=2).astype(bool) & others
        contact = near.sum() / perim
        dt = cv2.distanceTransform((~others).astype(np.uint8), cv2.DIST_L2, 3)
        pts = st["c"][:, 0, :]
        free_pts = pts[dt[pts[:, 1], pts[:, 0]] > 4]
    occluded = contact > 0.06
    c_score = float(np.clip(np.median(dist[mask]) / 40, 0, 1))
    out = dict(label="object", L=0.40, color=color, rgb=rgbv, evidence=[], attrs={"occluded": bool(occluded)}, bbox=None, st=st)

    if flat_std < FLAT_FILL_STD:                         # flat fill -> geometric shape
        out["evidence"].append(f"flat colour fill (Lab std {flat_std:.1f}) -> geometric shape")
        hull = st["hull"]
        approx = cv2.approxPolyDP(hull, 0.03 * cv2.arcLength(hull, True), True)[:, 0, :]
        if not occluded:
            if st["circ"] > 0.88 and len(approx) >= 6:
                out.update(label="circle", L=0.95); out["evidence"].append(f"circularity {st['circ']:.2f}")
            elif len(approx) == 3:
                out.update(label="triangle", L=0.95); out["evidence"].append("3 corners")
            elif len(approx) == 4 and st["rect_fill"] > 0.88:
                sq = st["aspect"] > 0.88
                out.update(label="square" if sq else "rectangle", L=0.95)
                out["evidence"].append(f"4 corners, aspect {st['aspect']:.2f}")
            else:
                out.update(label="shape", L=0.45); out["evidence"].append("flat region, no standard shape fits")
        else:
            out["evidence"].append(f"partly hidden behind another shape ({contact:.0%} of outline shared)")
            done = False
            if free_pts is not None and len(free_pts) > 25:
                cx, cy, r, res, ext = _fit_circle(free_pts)
                if res < 0.025 and ext > 1.5:
                    out.update(label="circle", L=0.85)
                    out["bbox"] = (int(cx - r), int(cy - r), int(2 * r), int(2 * r))
                    out["attrs"]["amodal_circle"] = [round(cx, 1), round(cy, 1), round(r, 1)]
                    out["evidence"].append(f"visible arc fits a circle (r={r:.0f}px, residual {res:.3f}); full extent restored")
                    done = True
            if not done and len(approx) == 4:
                q = approx.astype(float)
                e = [q[(i + 1) % 4] - q[i] for i in range(4)]
                p02, p13 = _parallel(e[0], e[2]), _parallel(e[1], e[3])
                if p02 and p13:
                    out.update(label="square" if st["aspect"] > 0.88 else "rectangle", L=0.80)
                    out["evidence"].append("two pairs of parallel sides")
                    done = True
                elif p02 != p13:
                    i0 = 1 if p02 else 0                       # the converging (leg) pair
                    apex = _line_intersection(q[i0], q[(i0 + 1) % 4], q[i0 + 2], q[(i0 + 3) % 4])
                    out.update(label="triangle", L=0.75)
                    out["evidence"].append("trapezoid with converging sides -> triangle whose apex is hidden")
                    if apex is not None and np.all(np.isfinite(apex)):
                        x0, y0, w0, h0 = cv2.boundingRect(st["c"])
                        nx0, ny0 = min(x0, apex[0]), min(y0, apex[1])
                        nx1, ny1 = max(x0 + w0, apex[0]), max(y0 + h0, apex[1])
                        if (nx1 - nx0) * (ny1 - ny0) < 3 * w0 * h0:
                            out["bbox"] = (int(nx0), int(ny0), int(nx1 - nx0), int(ny1 - ny0))
                            out["attrs"]["apex_restored"] = True
                    done = True
            if not done:
                out.update(label="shape", L=0.45)
    else:                                                # shaded / textured -> organic
        out["evidence"].append(f"shaded/textured surface (Lab std {flat_std:.1f})")
        h, s, v = rgb_to_hsv_deg(rgbv)
        round_ok = st["hull_circ"] > 0.85 and st["solidity"] > 0.90     # hull-based: a notch (apple lobes) must not count
        has_stem = any(p[0] in ("green", "brown") for p in regions_parts)
        if round_ok and (h < 28 or h >= 340):
            out.update(label="apple", L=0.80 + (0.08 if has_stem else 0.0))
            out["evidence"] += [f"round (hull circularity {st['hull_circ']:.2f}), red hue {h:.0f} deg"] + (["stem/leaf-coloured parts attached"] if has_stem else [])
        elif round_ok and 28 <= h < 50:
            out.update(label="orange", L=0.80)
            out["evidence"].append(f"round (hull circularity {st['hull_circ']:.2f}), orange hue {h:.0f} deg")
        elif round_ok:
            out.update(label="round object", L=0.40)
            out["evidence"].append("round shaded object, no fruit colour match")
        else:
            out.update(label="object", L=0.40)
            out["evidence"].append("irregular shaded region")
    q_score = float(np.clip(st["solidity"], 0, 1)) if not occluded else 0.9
    out["conf"] = round(min(0.99, out["L"] * (0.6 + 0.2 * c_score + 0.2 * q_score)), 2)
    return out


def _analyze_component(rgb, lab, comp, dist) -> list[Detection]:
    regions = _color_regions(rgb, lab, comp)
    if not regions:
        return []
    bb = mask_bbox(comp)
    fig = _try_figure(comp, regions, bb, dist)
    if fig:
        return [fig]
    amax = max(r.area for r in regions)
    indep = [r for r in regions if r.area >= 0.25 * amax]
    masks = [r.mask.copy() for r in indep]
    for p in (r for r in regions if r.area < 0.25 * amax):          # small parts (stem, leaf, eyes, AA fringe)
        dil = cv2.dilate(p.mask.astype(np.uint8), np.ones((5, 5), np.uint8)).astype(bool)
        contact = [int((dil & m).sum()) for m in masks]
        j = int(np.argmax(contact)) if max(contact) > 0 else int(np.argmin([(r.cx - p.cx) ** 2 + (r.cy - p.cy) ** 2 for r in indep]))
        masks[j] |= p.mask
        indep[j].parts.append((p.color, p.area))
    full = _partition(comp, masks)
    lab_core = lambda m: cv2.erode(m.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool)
    dets = []
    for i, m in enumerate(full):
        others = np.logical_or.reduce([full[j] for j in range(len(full)) if j != i]) if len(full) > 1 else None
        core = lab_core(m)
        core = core if core.sum() > 30 else m
        flat_std = float(np.max(np.std(lab[core], axis=0)))
        res = _classify_region(rgb, m, others, indep[i].parts, flat_std, dist)
        bbox = res["bbox"] or mask_bbox(m)
        d = Detection(res["label"], res["color"], res["conf"], tuple(int(v) for v in bbox), m, "classical", res["attrs"], res["evidence"])
        d.attrs["rgb"] = list(res["rgb"])
        if indep[i].parts:
            d.attrs["parts"] = sorted({p[0] for p in indep[i].parts})
        dets.append(d)
    # a row/grid of equal flat rectangles touching each other = colour swatches
    if len(dets) >= 3 and all(d.label in ("rectangle", "square") for d in dets):
        areas = np.array([d.mask.sum() for d in dets], float)
        if areas.std() / areas.mean() < 0.2:
            for d in dets:
                d.label = "color swatch"
                d.evidence.append(f"{len(dets)} touching equal-size rectangles -> colour chart")
    return dets


# ----------------------------------------------------------------------------- routes
def _classical(bgr, rgb, lab, bg) -> list[Detection]:
    H, W = lab.shape[:2]
    dist = np.linalg.norm(cv2.GaussianBlur(lab, (0, 0), 1.0) - bg, axis=2)
    fg = (dist > FG_THRESHOLD).astype(np.uint8)
    fg = cv2.morphologyEx(fg, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    fg = cv2.morphologyEx(fg, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    fg = fill_holes(fg).astype(np.uint8)
    n, cl, st, _ = cv2.connectedComponentsWithStats(fg, connectivity=8)
    min_area = max(60, int(0.002 * H * W))
    ids = [i for i in range(1, n) if st[i, cv2.CC_STAT_AREA] >= min_area]
    area = {i: int(st[i, cv2.CC_STAT_AREA]) for i in ids}
    masks = {i: cl == i for i in ids}
    # small fragments sitting right next to a much bigger object (apple leaf/stem) belong to it
    for small in sorted(ids, key=area.get):
        for big in sorted(ids, key=area.get, reverse=True):
            if big == small or small not in masks or big not in masks or area[small] >= 0.06 * area[big]:
                continue
            dt = cv2.distanceTransform((~masks[big]).astype(np.uint8), cv2.DIST_L2, 3)
            if dt[masks[small]].min() < 0.12 * np.sqrt(area[big]):
                masks[big] = masks[big] | masks.pop(small)
                break
    dets = []
    for i, m in masks.items():
        dets += _analyze_component(rgb, lab, m, dist)
    return dets


WEAK_SCORE = 0.30            # below this a proposal is only a suggestion ("possible"), never a confident result
MAX_PHOTO_OBJECTS = 14


def _contained(inner, outer, frac=0.8) -> bool:
    ix, iy, iw, ih = inner; ox, oy, ow, oh = outer
    w = max(0, min(ix + iw, ox + ow) - max(ix, ox)); h = max(0, min(iy + ih, oy + oh) - max(iy, oy))
    return w * h >= frac * iw * ih


def _proposals(bgr, yolox, conf, extra_labels):
    """Merge every available proposal source into one list of {'label','score','bbox','src'}."""
    from .yoloworld import YOLOWORLD
    props = [{**r, "src": "yolox"} for r in detect_tiled(yolox, bgr, conf=conf)] if yolox.available() else []
    ov_used = False
    
    # Fast open-vocabulary queries with YOLO-World when extra_labels requested
    if YOLOWORLD.available():
        labels = extra_labels or sorted(set(COCO_OF) | set(ROUND_CLASSES) | {"apple", "orange", "circle", "square", "triangle", "rectangle"})
        yw_results, _ = YOLOWORLD.detect(bgr, classes=labels, conf_threshold=conf)
        for y in yw_results:
            y["src"] = "yolo-world"
            props.append(y)
            ov_used = True

    # OWLv2 text-guided proposals (used when extra_labels are requested)
    if OPENVOCAB.available() and extra_labels:
        ov = OPENVOCAB.detect(bgr, extra=extra_labels)
        if ov:
            ov_used = True
        for o in ov:
            o["src"] = "open-vocab"
            # same physical object seen by both: keep the higher-scoring opinion, remember the other as a note
            clash = next((p for p in props if iou(p["bbox"], o["bbox"]) > 0.5), None)
            if clash is None:
                props.append(o)
            elif o["score"] >= clash["score"] * 0.8 and o["label"] != clash["label"].replace("_", " "):
                o["alt"] = f"{clash['label'].replace('_', ' ')} ({clash['score']:.2f}, YOLOX)"
                props.remove(clash); props.append(o)
    props = nms_items(props, 0.6)
    # a weak box that sits inside a stronger one is usually a part (keyboard inside laptop), not a separate object
    props = [p for p in props if not (p["score"] < 0.45 and any(q is not p and q["score"] > p["score"] and _contained(p["bbox"], q["bbox"]) for q in props))]
    return props[:MAX_PHOTO_OBJECTS], ov_used


def _neural(bgr, rgb, yolox: YoloxDetector, conf=0.15, extra_labels=None) -> tuple[list[Detection], bool]:
    dets = []
    props, ov_used = _proposals(bgr, yolox, conf, extra_labels)
    for r in props:
        label = r["label"].replace("_", " ")
        weak = r["score"] < WEAK_SCORE
        mask, q = grabcut_mask(bgr, r["bbox"])
        if q < 0.2:
            mask, q = box_mask(bgr.shape, r["bbox"]), 0.2
        if label in ROUND_CLASSES:                       # rescue shadowed sides of convex objects
            cnts, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            hull = cv2.convexHull(np.vstack(cnts))
            hm = np.zeros(mask.shape, np.uint8)
            cv2.fillPoly(hm, [hull], 1)
            mask = hm.astype(bool)
        shape_note = None
        roundish = label in ROUND_CLASSES or label in ("cup", "bottle", "wine glass", "mug", "ball")
        if q < (0.65 if roundish else 0.5):              # GrabCut caught only part of the object (e.g. the coffee, not the cup)
            x0, y0, bw0, bh0 = r["bbox"]
            if roundish:                                 # a whole round object fills about 78% (pi/4) of its box
                fm = np.zeros(mask.shape, np.uint8)
                cv2.ellipse(fm, (x0 + bw0 // 2, y0 + bh0 // 2), (max(1, bw0 // 2), max(1, bh0 // 2)), 0, 0, 360, 1, -1)
                mask, shape_note = fm.astype(bool), "ellipse"
            else:
                mask, shape_note = box_mask(bgr.shape, r["bbox"]), "box"
            q = max(q, 0.5)
        name, rgbv = region_color(rgb, mask, erode=9)
        h, _, _ = rgb_to_hsv_deg(rgbv)
        who = "open-vocabulary model" if r["src"] == "open-vocab" else "YOLOX-nano"
        evid = [f"{who} proposes '{label}' ({r['score']:.2f})", f"GrabCut mask covers {q:.0%} of the box"]
        if shape_note:
            evid.append(f"the segmentation covered too little of the box, so the outline falls back to the model's {shape_note}")
        if r.get("alt"):
            evid.append(f"note: the other detector said {r['alt']}")
        if weak:
            evid.append("weak proposal: shown as a suggestion, please confirm or remove it")
        agree = 1.0
        if r["src"] == "yolox" and label in ("apple", "orange"):
            hue_label = "apple" if (h < 30 or h >= 340) else ("orange" if h < 50 else None)
            if hue_label and hue_label != label:
                evid.append(f"colour check: hue {h:.0f} deg is {'red-orange' if hue_label == 'apple' else 'yellow-orange'}"
                            f" -> relabelled {label} -> {hue_label} (nano-sized detectors confuse these two fruits)")
                label, agree = hue_label, 0.5
            else:
                evid.append(f"colour check agrees (hue {h:.0f} deg)")
        conf_ = round(min(0.95, 0.55 * r["score"] + 0.30 * agree + 0.15 * min(1.0, q / 0.6)), 2)
        if weak:
            conf_ = min(conf_, 0.55)                      # always lands under the review threshold
        bbox = r["bbox"] if shape_note else (mask_bbox(mask) if q >= 0.35 else r["bbox"])
        dets.append(Detection(label, name, conf_, tuple(bbox), mask, "neural" if r["src"] == "yolox" else "open-vocab",
                              {"neural_score": round(r["score"], 3), "rgb": list(rgbv), "source": r["src"], "weak": weak}, evid))
    return dets, ov_used


COCO_OF = {"boy": "person", "girl": "person", "apple": "apple", "orange": "orange"}
GENERIC_LABELS = {"object", "round object", "shape"}


def _corroborate(dets, bgr, yolox: YoloxDetector):
    raw = yolox.detect(bgr, conf=0.20)
    for d in dets:
        want = COCO_OF.get(d.label)
        best_match = None
        best_iou = 0.0
        for r in raw:
            match_iou = iou(d.bbox, r["bbox"])
            if match_iou > best_iou:
                best_iou = match_iou
                best_match = r
        if best_match and best_iou >= 0.35:
            r = best_match
            lbl = r["label"].replace("_", " ")
            if want and r["label"] == want:
                d.confidence = round(1 - (1 - d.confidence) * (1 - 0.7 * r["score"]), 2)
                d.evidence.append(f"YOLOX-nano independently agrees: '{r['label']}' {r['score']:.2f}")
                d.attrs["neural_agrees"] = True
            elif d.label in GENERIC_LABELS and r["score"] >= 0.25:
                # Upgrade generic label (e.g. "object") to recognized COCO class (e.g. "cat", "dog", "bottle", "scissors")
                old_label = d.label
                d.label = lbl
                d.confidence = round(max(d.confidence, float(r["score"])), 2)
                d.evidence.append(f"Neural detector identified '{old_label}' as '{d.label}' ({r['score']:.2f})")
                d.attrs["neural_agrees"] = True
            elif r["score"] > 0.5 and lbl != d.label:
                d.evidence.append(f"note: YOLOX-nano suggests '{lbl}' ({r['score']:.2f})")


# ----------------------------------------------------------------------------- public API
_YOLOX = YoloxDetector()


def detect(bgr: np.ndarray, use_neural: bool = True, extra_labels: list | None = None, conf_threshold: float | None = None) -> DetectionResult:
    t0 = time.time()
    H, W = bgr.shape[:2]
    s = min(1.0, 1024 / max(H, W))
    work = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else bgr
    rgb = cv2.cvtColor(work, cv2.COLOR_BGR2RGB)
    lab = to_lab(work)
    bg, spread = border_background(lab)
    neural_ok = use_neural and _YOLOX.available()
    flat = spread < FLAT_BG_SPREAD
    msg = ""
    req_conf = conf_threshold if conf_threshold is not None else 0.15
    if flat:
        route = "flat-background"
        dets = _classical(work, rgb, lab, bg)
        from .yoloworld import YOLOWORLD
        if YOLOWORLD.available():
            world, _ = _neural(work, rgb, _YOLOX, conf=req_conf, extra_labels=extra_labels)
            for candidate in world:
                if not any(iou(candidate.bbox, existing.bbox) >= 0.5 for existing in dets):
                    dets.append(candidate)
            if world:
                route = "flat-background + YOLO-World"
        if dets and neural_ok:
            _corroborate(dets, work, _YOLOX)
        if not dets and neural_ok:
            dets, _ = _neural(work, rgb, _YOLOX, conf=max(req_conf, 0.30), extra_labels=extra_labels)
            route = "flat-background -> neural second opinion"
        if not dets:
            msg = "Nothing stands out from the background (object may be too faint or absent)."
    else:
        route = "complex-background"
        if neural_ok:
            dets, ov_used = _neural(work, rgb, _YOLOX, conf=req_conf, extra_labels=extra_labels)
            route = "complex-background + open-vocabulary" if (extra_labels or ov_used) else "complex-background"
            if not dets:
                msg = "No known object found on this busy background."
        else:
            dets, msg = [], "Busy background and the neural backend is unavailable (run scripts/fetch_assets.py)."
    if s < 1:                                             # bring results back to original resolution
        for d in dets:
            d.mask = cv2.resize(d.mask.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST).astype(bool)
            x, y, w, h = d.bbox
            d.bbox = (int(x / s), int(y / s), int(w / s), int(h / s))
    dets.sort(key=lambda d: (d.bbox[1] // 40, d.bbox[0]))
    if not dets:
        status = "failed" if (not flat and not neural_ok) else "no_objects"
    elif min(d.confidence for d in dets) < NEEDS_REVIEW:
        status = "low_confidence"
    else:
        status = "ok"
    union = np.logical_or.reduce([d.mask for d in dets]) if dets else None
    colors = dominant_color_names(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), union, 4)
    return DetectionResult(status, dets, route, msg, (time.time() - t0) * 1000, colors)
