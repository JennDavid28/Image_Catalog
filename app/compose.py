"""Composite image generation from catalog assets.

Pipeline:  instruction --parse--> plan --resolve--> catalog assets --layout--> placements --render--> PNG

Design rules
  * The composer ONLY uses pixels cut out of catalog objects. If a requested item has no matching asset the request
    fails with an explicit list of what is missing (+ close matches) - it never draws a stand-in.
  * Parsing is deterministic rules by default (fast, testable, offline). An LLM can be used to *propose* the plan
    (see llm_plan) but its output goes through the same validation + resolver, so it cannot invent assets.
  * Colour/label constraints are respected ("red apple" will not silently use an orange).
"""
from __future__ import annotations
import difflib
import json
import os
import re
import urllib.request
from dataclasses import dataclass, field

import cv2
import numpy as np

# ----------------------------------------------------------------------------- vocabulary
COLOR_WORDS = {"red", "orange", "yellow", "green", "blue", "purple", "pink", "brown", "black", "white", "gray", "grey",
               "cyan", "peach", "violet"}
COLOR_ALIAS = {"grey": "gray", "violet": "purple"}
NUMBERS = {"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8,
           "nine": 9, "ten": 10, "pair": 2, "couple": 2, "several": 3, "few": 3}
SIZE_WORDS = {"tiny": 0.5, "small": 0.7, "little": 0.7, "mini": 0.6, "medium": 1.0, "big": 1.35, "large": 1.35,
              "huge": 1.7, "giant": 1.8}
CATEGORIES = {                       # plural / category noun -> labels it stands for
    "shape": {"circle", "square", "triangle", "rectangle", "shape"},
    "fruit": {"apple", "orange", "banana", "pear", "lemon", "grape"},
    "person": {"boy", "girl", "person", "man", "woman", "child", "kid"},
    "color swatch": {"color swatch"},
}
SYNONYMS = {"swatch": "color swatch", "color": "color swatch", "colour swatch": "color swatch", "colour": "color swatch",
            "child": "person", "kid": "person", "people": "person", "human": "person", "ball": "circle", "disc": "circle",
            "box": "square", "tri": "triangle", "rect": "rectangle"}
STOP = {"the", "a", "an", "of", "some", "set", "pair", "couple", "few", "several", "colored", "coloured", "colorful",
        "colourful", "multicolored", "different", "image", "picture", "composite", "scene", "illustration", "collage",
        "separate", "individual", "single", "just", "only", "exactly", "two-dimensional", "cartoon", "cartoonish"}
GROUP_NOUNS = {"shape": "shape", "fruit": "fruit", "people": "person", "swatch": "color swatch", "color": "color swatch",
               "colour": "color swatch", "kid": "person", "child": "person", "children": "person"}

RELATIONS = [                         # (regex, relation of FIRST item w.r.t. SECOND item) - longest phrases first
    (r"to the left of|on the left of|left of|to the left side of", "left_of"),
    (r"to the right of|on the right of|right of|to the right side of", "right_of"),
    (r"on top of|above|over", "above"),
    (r"underneath|beneath|below|under", "below"),
    (r"in front of|before", "in_front"),
    (r"behind|in back of", "behind"),
    (r"next to|beside|alongside|near|close to", "adjacent"),
    (r"overlapping with|overlapping|partly covering|partially covering", "in_front"),
    (r",?\s*and\b|,|\bplus\b|&|\bwith\b|\balong with\b|\btogether with\b", "adjacent"),
]
_REL_RE = re.compile("(" + "|".join(f"(?:{p})" for p, _ in RELATIONS) + ")")
ANCHOR_RE = re.compile(r"\b(?:on|at|in|to) the (left|right|top|bottom|middle|center|centre)(?: side)?(?: of the (?:image|picture|canvas|frame))?\b")
BG_RE = re.compile(r"\b(?:on|against|over|with|in front of)\s+(?:an?|the)?\s*([a-zA-Z0-9_\s-]+?)\s*(?:background|backdrop)\b", re.I)
PREFIX_RE = re.compile(r"^\s*(?:please\s+)?(?:can you\s+)?(?:create|make|generate|draw|compose|build|produce|render|give me|show me|place|put|add|position|set)"
                       r"(?:\s+me)?\s*(?:an?|the)?\s*(?:new\s+)?(?:image|picture|composite|scene|illustration|collage|composition)?"
                       r"\s*(?:image)?\s*(?:containing|that contains|with|of|that has|showing|featuring|including|made of|using)?\s*", re.I)

BG_COLORS = {"white": (255, 255, 255), "black": (20, 20, 24), "gray": (200, 200, 205), "red": (235, 120, 120),
             "green": (160, 215, 170), "blue": (170, 200, 240), "yellow": (250, 240, 170), "orange": (250, 205, 150),
             "purple": (210, 185, 235), "pink": (250, 205, 220), "beige": (240, 230, 210), "cream": (250, 245, 225),
             "brown": (190, 160, 130), "cyan": (170, 230, 240), "lightblue": (190, 220, 245)}
HEIGHT_FRAC = {"person": 0.72, "fruit": 0.27, "shape": 0.19, "color swatch": 0.17}   # target height / canvas height


class CompositionError(Exception):
    def __init__(self, message, missing=None, suggestions=None, plan=None):
        super().__init__(message)
        self.missing, self.suggestions, self.plan = missing or [], suggestions or {}, plan


# ----------------------------------------------------------------------------- text normalisation
def singular(w: str) -> str:
    if w in ("people", "children"):
        return {"people": "person", "children": "child"}[w]
    if w.endswith("ies") and len(w) > 4:
        return w[:-3] + "y"
    if re.search(r"(ch|sh|x|ss|z)es$", w):
        return w[:-2]
    if w.endswith("ses") and len(w) > 4:
        return w[:-2]
    if w.endswith("s") and not w.endswith("ss") and len(w) > 3:
        return w[:-1]
    return w


def norm_label(label: str) -> str:
    return " ".join(singular(t) for t in re.findall(r"[a-z]+", (label or "").lower()))


def category_of(noun: str):
    for cat, members in CATEGORIES.items():
        if noun == cat or noun in members:
            return cat
    return None


def label_matches(noun: str, label: str) -> bool:
    """noun is canonical (singular); label is a catalog label such as 'apple', 'green triangle', 'color swatch'."""
    l = norm_label(label)
    if not l or not noun:
        return False
    if l == noun or l.endswith(" " + noun):
        return True
    cat = category_of(noun)
    if cat and noun == cat:                                   # category noun ('shape', 'fruit')
        return l in CATEGORIES[cat] or any(l.endswith(" " + m) for m in CATEGORIES[cat])
    return False


# ----------------------------------------------------------------------------- parsing
@dataclass
class Item:
    id: str
    raw: str
    noun: str
    count: int = 1
    color: str | None = None
    size: float = 1.0
    anchor: str | None = None
    group: bool = False                 # category plural ("colored shapes") -> one best asset per distinct label
    relation: str = "adjacent"          # relation of the PREVIOUS item with respect to this one


def parse_instruction(text: str) -> dict:
    """Rule-based NL -> plan. Returns {'canvas':..., 'items':[Item...], 'warnings':[...]}."""
    warnings = []
    t = " " + re.sub(r"\s+", " ", text.lower().strip().rstrip(".!")) + " "
    t = PREFIX_RE.sub(" ", t.strip(), count=1)
    t = " " + t + " "
    bg = "white"
    transparent = bool(re.search(r"\btransparent\b", t))
    if transparent:
        t = re.sub(r"\b(?:on|with|a|an|the|in)?\s*(?:a|an|the)?\s*transparent\s*(?:background|backdrop)?\b", " ", t)
    m = BG_RE.search(t)
    if m and not transparent:
        bg_desc = (m.group(1) or "").strip().lower()
        words = bg_desc.split()
        shade = words[0] if (words and words[0] in ("light", "pale", "soft", "dark")) else ""
        col = words[-1] if words else ""
        if col in BG_COLORS or col in COLOR_ALIAS:
            col = COLOR_ALIAS.get(col, col)
            bg = ("light" + col) if (shade in ("light", "pale", "soft") and ("light" + col) in BG_COLORS) else col
            t = t[:m.start()] + " " + t[m.end():]
        elif bg_desc:
            bg = bg_desc
            t = t[:m.start()] + " " + t[m.end():]
    parts = _REL_RE.split(t)
    chunks, rels = [], []
    pending_rel = "adjacent"
    for i, piece in enumerate(parts):
        if i % 2 == 0:
            if piece and piece.strip(" ,"):
                chunks.append(piece.strip(" ,"))
                rels.append(pending_rel)
                pending_rel = "adjacent"
        else:
            for pat, rel in RELATIONS:
                if re.fullmatch(pat, piece.strip()) or re.fullmatch(pat, piece):
                    # specific spatial words override a plain 'and'
                    pending_rel = rel if (rel != "adjacent" or pending_rel == "adjacent") else pending_rel
                    break
    items: list[Item] = []
    for idx, ch in enumerate(chunks):
        it = _parse_chunk(ch, f"i{idx + 1}")
        if it is None:
            warnings.append(f"ignored phrase '{ch}' (no object found)")
            continue
        it.relation = rels[idx] if items else "adjacent"
        items.append(it)
    return {"canvas": {"background": "transparent" if transparent else bg}, "items": items, "warnings": warnings}


def _parse_chunk(chunk: str, item_id: str) -> Item | None:
    anchor = None
    m = ANCHOR_RE.search(chunk)
    if m:
        anchor = {"centre": "center", "middle": "center"}.get(m.group(1), m.group(1))
        chunk = (chunk[:m.start()] + " " + chunk[m.end():]).strip()
    toks = re.findall(r"[a-z0-9]+", chunk)
    count, color, size, nouns, explicit = 1, None, 1.0, [], False
    i = 0
    while i < len(toks):
        w = toks[i]
        if w.isdigit():
            count, explicit = max(1, min(int(w), 12)), True
        elif w in NUMBERS and w not in ("pair", "couple", "few", "several"):
            if w not in ("a", "an"):
                explicit = True
            count = NUMBERS[w]
        elif w in ("pair", "couple", "few", "several"):
            count, explicit = NUMBERS[w], True
        elif w in SIZE_WORDS:
            size = SIZE_WORDS[w]
        elif w in COLOR_WORDS:
            nouns.append(("color", COLOR_ALIAS.get(w, w)))
        elif w in STOP or w in ("in", "on", "at", "to", "that", "is", "are", "be", "it", "its", "their", "this", "these", "those", "and"):
            pass
        else:
            nouns.append(("word", w))
        i += 1
    words = [v for k, v in nouns if k == "word"]
    colors = [v for k, v in nouns if k == "color"]
    if not words:
        if colors:                                            # "one orange": the only word is a colour name -> treat as the noun
            words, colors = [colors[-1]], colors[:-1]
        else:
            return None
    noun_raw = " ".join(words)
    # try the last 2 words as a compound label ('color swatch'), else last word
    cand = [norm_label(noun_raw), norm_label(words[-1])]
    noun = None
    for c in cand:
        c = SYNONYMS.get(c, c)
        if c:
            noun = c
            break
    plural_group = False
    last = words[-1]
    if singular(last) != last or last in ("people", "children"):
        if explicit and count > 1 and category_of(noun) == noun and noun in CATEGORIES:
            plural_group = True
        elif not explicit and (category_of(noun) == noun) and noun in CATEGORIES:
            plural_group = True
        elif not explicit:                                    # "apples" with no number -> keep count 2
            count = max(count, 2)
    if noun in CATEGORIES and not explicit and last in ("shapes", "fruits", "swatches", "colors", "colours", "people", "kids", "children"):
        plural_group = True
    color = colors[-1] if colors else None
    return Item(item_id, chunk, noun, count, color, size, anchor, plural_group)


# ----------------------------------------------------------------------------- resolution (plan -> catalog assets)
def _rank_key(o: dict, n_in_image: dict):
    verified = o["origin"] == "manual" or o["edited"] or o["reviewed"]
    return (verified, o["confidence"] or 0.0, -n_in_image.get(o["image_id"], 1), o["image_kind"] == "source")


ACHROMATIC = {"white", "gray", "light gray", "black"}
SHAPE_ORDER = ["circle", "square", "triangle", "rectangle"]


def base_noun(label: str) -> str:
    """'light gray circle' -> 'circle' (so a user's custom name does not count as a new kind of shape)."""
    l = norm_label(label)
    last = l.split()[-1] if l else l
    return last if any(last in m for m in CATEGORIES.values()) else l


def _chromatic(o: dict) -> bool:
    return any(part.strip() not in ACHROMATIC for part in re.split(r"[/,]", (o["color"] or "").lower()) if part.strip())


def _color_ok(o: dict, color: str | None) -> bool:
    if not color:
        return True
    c = (o["color"] or "").lower()
    return color in re.split(r"[/ ,]+", c) or c == color or c.endswith(" " + color)


def resolve(items: list[Item], objects: list[dict]):
    """Returns (resolved, missing, suggestions). resolved: list of dicts {item, assets:[object dicts]}."""
    n_in_image: dict[str, int] = {}
    for o in objects:
        n_in_image[o["image_id"]] = n_in_image.get(o["image_id"], 0) + 1
    labels = sorted({norm_label(o["label"]) for o in objects})
    resolved, missing, suggestions = [], [], {}
    for it in items:
        cands = [o for o in objects if label_matches(it.noun, o["label"])]
        if it.color:
            filt = [o for o in cands if _color_ok(o, it.color)]
            if not filt and cands:
                have = sorted({(o["color"] or "?") for o in cands})
                missing.append(f"'{it.raw}': catalog has {it.noun} but not in {it.color} (available: {', '.join(have)})")
                continue
            cands = filt
        if not cands:
            missing.append(f"'{it.raw}': no catalog object labelled '{it.noun}'")
            close = difflib.get_close_matches(it.noun, labels, n=3, cutoff=0.5)
            if close:
                suggestions[it.raw] = close
            continue
        if it.group:                                          # one best asset per distinct label
            # one asset per KIND (base noun); for "colored ..." requests a coloured asset beats a grey/white one
            best: dict[str, dict] = {}
            for o in sorted(cands, key=lambda o: (_chromatic(o), *_rank_key(o, n_in_image)), reverse=True):
                best.setdefault(base_noun(o["label"]), o)
            assets = sorted(best.values(), key=lambda o: SHAPE_ORDER.index(base_noun(o["label"])) if base_noun(o["label"]) in SHAPE_ORDER else 99)
            if it.count > 1 and it.count < len(assets):
                assets = assets[:it.count]
            resolved.append({"item": it, "assets": assets})
        else:
            top = max(cands, key=lambda o: _rank_key(o, n_in_image))
            resolved.append({"item": it, "assets": [top] * it.count})
    return resolved, missing, suggestions


# ----------------------------------------------------------------------------- layout
@dataclass
class Leaf:
    item_id: str
    asset: dict
    w: float
    h: float
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


@dataclass
class Node:
    item_id: str
    leaves: list                        # Leaf with coordinates relative to node origin
    w: float
    h: float
    relation: str
    anchor: str | None
    x: float = 0.0
    y: float = 0.0
    z: float = 0.0


def _target_h(asset: dict, size: float, H: int) -> float:
    cat = category_of(norm_label(asset["label"].split()[-1])) or category_of(norm_label(asset["label"]))
    if cat in HEIGHT_FRAC:
        frac = HEIGHT_FRAC[cat]
    else:                                                    # unknown label: keep its proportion within the source image
        frac = float(np.clip(asset["bbox"][3] / max(1, asset["image_size"][1]), 0.12, 1.0)) * 0.5
    return frac * H * size


def _leaf(item_id: str, asset: dict, size: float, H: int) -> Leaf:
    th = _target_h(asset, size, H)
    ar = asset["bbox"][2] / max(1, asset["bbox"][3])
    return Leaf(item_id, asset, th * ar, th)


def build_nodes(resolved, H: int) -> list[Node]:
    nodes = []
    for r in resolved:
        it: Item = r["item"]
        leaves = [_leaf(it.id, a, it.size, H) for a in r["assets"]]
        gap = 0.025 * H
        if len(leaves) == 1:
            node_leaves, w, h = leaves, leaves[0].w, leaves[0].h
        elif it.group and len(leaves) >= 4:                   # compact 2-column grid
            rows = [leaves[i:i + 2] for i in range(0, len(leaves), 2)]
            col_w = [max(l.w for l in (r_[c] for r_ in rows if len(r_) > c)) for c in range(2)]
            y = 0.0
            for r_ in rows:
                rh = max(l.h for l in r_)
                x = 0.0
                for c, l in enumerate(r_):
                    l.x, l.y = x + (col_w[c] - l.w) / 2, y + (rh - l.h) / 2
                    x += col_w[c] + gap
                y += rh + gap
            node_leaves, w, h = leaves, sum(col_w) + gap, y - gap
        else:                                                 # row, bottoms aligned
            x = 0.0
            hmax = max(l.h for l in leaves)
            for l in leaves:
                l.x, l.y = x, hmax - l.h
                x += l.w + gap
            node_leaves, w, h = leaves, x - gap, hmax
        nodes.append(Node(it.id, node_leaves, w, h, it.relation, it.anchor))
    return nodes


def layout(nodes: list[Node], W: int, H: int) -> list[Leaf]:
    gap = 0.045 * H
    if any(n.anchor for n in nodes):
        return _layout_anchored(nodes, W, H)
    for k, n in enumerate(nodes):
        n.z = float(k)
        if k == 0:
            continue
        a = nodes[k - 1]
        rel = n.relation                                       # relation of A (previous) with respect to n
        if rel in ("adjacent", "left_of"):
            n.x, n.y = a.x + a.w + gap, a.y + a.h - n.h
        elif rel == "right_of":
            n.x, n.y = a.x - n.w - gap, a.y + a.h - n.h
        elif rel == "above":
            n.x, n.y = a.x + a.w / 2 - n.w / 2, a.y + a.h + gap * 0.5
        elif rel == "below":
            n.x, n.y = a.x + a.w / 2 - n.w / 2, a.y - n.h - gap * 0.5
        elif rel == "in_front":
            n.x, n.y, n.z = a.x + a.w * 0.62, a.y + a.h - n.h, a.z - 0.5
        elif rel == "behind":
            n.x, n.y, n.z = a.x + a.w * 0.62, a.y + a.h - n.h, a.z + 0.5
    x0, y0 = min(n.x for n in nodes), min(n.y for n in nodes)
    x1, y1 = max(n.x + n.w for n in nodes), max(n.y + n.h for n in nodes)
    gw, gh = x1 - x0, y1 - y0
    s = min(1.6, 0.80 * W / gw, 0.80 * H / gh)          # fit the whole group to ~80% of the canvas; relative sizes unchanged
    ox, oy = (W - gw * s) / 2 - x0 * s, (H - gh * s) / 2 - y0 * s
    return _flatten(nodes, s, ox, oy)


def _scale_node(n: Node, f: float):
    """Uniformly scale a node about its bottom-centre (it stays standing where it was)."""
    cx, bottom = n.x + n.w / 2, n.y + n.h
    for l in n.leaves:
        l.x, l.y, l.w, l.h = l.x * f, l.y * f, l.w * f, l.h * f
    n.w, n.h = n.w * f, n.h * f
    n.x, n.y = cx - n.w / 2, bottom - n.h


def _resolve_collisions(nodes: list[Node], W: int, margin: float):
    """Anchored placement can put a wide node (e.g. a grid of shapes) on top of its neighbours.
    Shrink the offender (the centre one, else the wider one) in small steps until the nodes clear each other."""
    for _ in range(40):
        moved = False
        order = sorted(nodes, key=lambda n: n.x)
        for a, b in zip(order, order[1:]):
            y_overlap = min(a.y + a.h, b.y + b.h) - max(a.y, b.y) > 0
            if y_overlap and a.x + a.w + margin > b.x:
                victim = next((n for n in (a, b) if n.anchor == "center"), a if a.w >= b.w else b)
                if victim.w < 0.35 * W * 0.5:                     # do not shrink below a sensible minimum
                    victim = b if victim is a else a
                _scale_node(victim, 0.94)
                moved = True
        if not moved:
            return


def _layout_anchored(nodes: list[Node], W: int, H: int) -> list[Leaf]:
    cx = {"left": 0.22, "center": 0.5, "right": 0.78}
    free = [0.5, 0.22, 0.78]
    for n in nodes:
        if n.anchor in cx:
            free = [f for f in free if abs(f - cx[n.anchor]) > 1e-6]
    for k, n in enumerate(nodes):
        n.z = float(k)
        fx = cx.get(n.anchor) if n.anchor in cx else None
        if fx is None:
            fx = free.pop(0) if free else 0.5
        cy = {"top": 0.28, "bottom": 0.72}.get(n.anchor)
        n.x = fx * W - n.w / 2
        n.y = (cy * H - n.h / 2) if cy is not None else (0.5 * H + 0.5 * max(x.h for x in nodes) - n.h)
    _resolve_collisions(nodes, W, margin=0.02 * W)
    return _flatten(nodes, 1.0, 0.0, 0.0)


def _flatten(nodes, s, ox, oy) -> list[Leaf]:
    out = []
    for n in nodes:
        for l in n.leaves:
            out.append(Leaf(l.item_id, l.asset, l.w * s, l.h * s, ox + (n.x + l.x) * s, oy + (n.y + l.y) * s, n.z))
    return out


# ----------------------------------------------------------------------------- rendering
def _resize_premult(bgra: np.ndarray, w: int, h: int) -> np.ndarray:
    a = bgra[..., 3:4].astype(np.float32) / 255.0
    pm = np.concatenate([bgra[..., :3].astype(np.float32) * a, a * 255.0], axis=2)
    interp = cv2.INTER_AREA if w < bgra.shape[1] else cv2.INTER_CUBIC
    pm = cv2.resize(pm, (max(1, w), max(1, h)), interpolation=interp)
    return pm                                                  # premultiplied BGR + alpha(0..255)


def render(leaves: list[Leaf], catalog, W: int, H: int, background: str | np.ndarray = "white", shadow: bool = True):
    """Alpha-composite catalog cut-outs (premultiplied 'over') onto the canvas. Returns (png_bytes, placements)."""
    transparent = isinstance(background, str) and background == "transparent"
    canvas = np.zeros((H, W, 4), np.float32)                  # premultiplied BGRA, 0..255
    if isinstance(background, np.ndarray):
        bg_canvas = cv2.resize(background[..., :3], (W, H), interpolation=cv2.INTER_AREA)
        canvas[..., :3] = bg_canvas.astype(np.float32)
        canvas[..., 3] = 255.0
    elif not transparent:
        bg = BG_COLORS.get(background, BG_COLORS["white"])
        canvas[..., :3] = np.array(bg[::-1], np.float32)
        canvas[..., 3] = 255.0
    placements = []
    for l in sorted(leaves, key=lambda l: l.z):
        asset = catalog.object_asset(l.asset["id"])
        w, h = max(1, int(round(l.w))), max(1, int(round(l.h)))
        pm = _resize_premult(asset, w, h)
        x, y = int(round(l.x)), int(round(l.y))
        if shadow:
            sh = np.zeros((H, W), np.float32)
            cv2.ellipse(sh, (x + w // 2, y + h), (max(2, int(w * 0.42)), max(2, int(h * 0.045))), 0, 0, 360, 1.0, -1)
            sh = cv2.GaussianBlur(sh, (0, 0), max(1.0, h * 0.02)) * (0.22 if not transparent else 0.12) * 255
            canvas[..., :3] *= (1 - sh[..., None] / 255.0)
            if transparent:
                canvas[..., 3] = np.maximum(canvas[..., 3], sh * 0.9)
        x0, y0, x1, y1 = max(0, x), max(0, y), min(W, x + w), min(H, y + h)
        if x1 <= x0 or y1 <= y0:
            continue
        src = pm[y0 - y:y1 - y, x0 - x:x1 - x]
        sa = src[..., 3:4] / 255.0
        dst = canvas[y0:y1, x0:x1]
        dst[..., :3] = src[..., :3] + dst[..., :3] * (1 - sa)
        dst[..., 3:4] = src[..., 3:4] + dst[..., 3:4] * (1 - sa)
        placements.append({"item_id": l.item_id, "asset_object_id": l.asset["id"], "asset_image_id": l.asset["image_id"],
                           "asset_image_name": l.asset["image_name"], "label": l.asset["label"], "color": l.asset["color"],
                           "bbox": [x, y, w, h], "z": l.z,
                           "scale": round(w / max(1, l.asset["bbox"][2]), 3),
                           "locked": "logo" in l.asset["label"].lower()})
    if transparent:
        a = np.clip(canvas[..., 3:4], 1e-3, 255.0)
        out = np.concatenate([np.clip(canvas[..., :3] / (a / 255.0), 0, 255), canvas[..., 3:4]], axis=2).astype(np.uint8)
    else:
        out = np.clip(canvas[..., :3], 0, 255).astype(np.uint8)
    ok, buf = cv2.imencode(".png", out)
    return buf.tobytes(), placements


# ----------------------------------------------------------------------------- public API
def plan(catalog, instruction: str, llm: bool = False, mode: str | None = None, user_id: str | None = None) -> dict:
    """Parse + resolve without rendering (the UI shows this as a preview)."""
    from .planner import plan_prompt
    effective_mode = mode

    parsed = None
    if llm:
        parsed = llm_plan(instruction, catalog, user_id=user_id)
    if parsed is None:
        parsed = parse_instruction(instruction)

    # Intelligent prompt understanding if requested
    llm_info = plan_prompt(instruction, catalog, user_id=user_id) if (llm or mode in ("hybrid", "ai_generated")) else {}
    if not effective_mode and llm_info.get("generation_mode"):
        effective_mode = llm_info["generation_mode"]
    effective_mode = effective_mode or "assets_only"

    objects = [o for o in catalog.live_objects(user_id) if o["image_kind"] == "source"]   # never reuse composites as assets
    resolved, missing, suggestions = resolve(parsed["items"], objects)
    ai_generated_elements = []
    if effective_mode == "hybrid":
        resolved_item_ids = {r["item"].id for r in resolved}
        ai_generated_elements = [it.raw for it in parsed["items"] if it.id not in resolved_item_ids]
        # In hybrid mode, missing objects are generated by AI rather than causing a blocking failure
        missing = []

    return {
        "instruction": instruction,
        "mode": effective_mode,
        "generation_mode": effective_mode,
        "canvas": parsed["canvas"],
        "warnings": parsed["warnings"],
        "planner": "llm" if (llm and parsed.get("_llm")) else ("intelligent_planner" if llm_info else "rules"),
        "planner_model": parsed.get("_planner") or llm_info.get("_planner"),
        "explanation": llm_info.get("explanation", ""),
        "clarification_needed": llm_info.get("clarification_needed", False),
        "clarification_question": llm_info.get("clarification_question"),
        "clarification_options": llm_info.get("clarification_options", []),
        "ai_generation": llm_info.get("ai_generation"),
        "ai_generated_elements": ai_generated_elements,
        "items": [{"id": it.id, "text": it.raw, "label": it.noun, "count": it.count, "color": it.color, "size": it.size,
                   "anchor": it.anchor, "group": it.group, "relation_to_previous": it.relation} for it in parsed["items"]],
        "resolved": [{"item": r["item"].id, "assets": [{"object_id": a["id"], "image_id": a["image_id"], "image_name": a["image_name"],
                                                        "label": a["label"], "color": a["color"], "confidence": a["confidence"],
                                                        "origin": a["origin"]} for a in r["assets"]]} for r in resolved],
        "missing": missing,
        "suggestions": suggestions,
        "_resolved": resolved,
        "_items": parsed["items"]
    }


def compose(catalog, instruction: str, width: int = 1024, height: int = 768, background: str | None = None,
            allow_partial: bool = False, name: str | None = None, llm: bool = False, shadow: bool = True,
            mode: str | None = None, ai_style: str = "realistic", user_id: str | None = None) -> dict:
    from .generator import IMAGE_GENERATOR

    p = plan(catalog, instruction, llm=llm, mode=mode, user_id=user_id)
    public = {k: v for k, v in p.items() if not k.startswith("_")}
    gen_mode = mode or p.get("generation_mode", "assets_only")

    # Mode 3: Pure AI Generation
    if gen_mode == "ai_generated":
        png, gen_meta = IMAGE_GENERATOR.generate_image(
            prompt=instruction,
            style=ai_style,
            width=width,
            height=height
        )
        provenance = {
            "instruction": instruction,
            "generation_mode": "ai_generated",
            "generator": gen_meta,
            "canvas": {"width": width, "height": height, "style": ai_style},
            "planner": p["planner"],
            "planner_model": p.get("planner_model")
        }
        # Add generated image to catalog and run automatic detection
        img = catalog.add_image(png, name or _auto_name(instruction), detect=True, kind="composite", provenance=provenance, user_id=user_id)
        return {
            "image": img,
            "plan": public,
            "placements": [],
            "warnings": p["warnings"],
            "mode": "ai_generated",
            "generation_mode": "ai_generated"
        }

    # Mode 2: Hybrid (Catalog Assets + AI Elements / Background)
    if gen_mode == "hybrid":
        if not p["_resolved"]:
            # If no catalog assets match at all, synthesize complete scene using AI
            png, gen_meta = IMAGE_GENERATOR.generate_image(
                prompt=instruction,
                style=ai_style,
                width=width,
                height=height
            )
            provenance = {
                "instruction": instruction,
                "generation_mode": "hybrid",
                "notice": "No catalog assets matched; synthesized entire scene using AI",
                "generator": gen_meta,
                "canvas": {"width": width, "height": height, "style": ai_style},
                "planner": p["planner"],
                "planner_model": p.get("planner_model")
            }
            img = catalog.add_image(png, name or _auto_name(instruction), detect=True, kind="composite", provenance=provenance, user_id=user_id)
            return {
                "image": img,
                "plan": public,
                "placements": [],
                "warnings": p["warnings"] + ["No matching catalog assets found; generated entire scene using AI elements"],
                "mode": "hybrid",
                "generation_mode": "hybrid"
            }

        # Some assets are in catalog, while remaining objects/background are synthesized by AI
        resolved_labels = [r["item"].noun for r in p["_resolved"]]
        ai_elements = [it.raw for it in p["_items"] if it.id not in {r["item"].id for r in p["_resolved"]}]
        if ai_elements:
            bg_prompt = f"A high-quality scene featuring clearly visible {' and '.join(ai_elements)}, on a clean surface or background setting, leaving space on the sides for additional objects"
        else:
            bg_prompt = f"A beautiful clean background scenery suitable for: {instruction}"

        bg_png, bg_meta = IMAGE_GENERATOR.generate_background(
            prompt=bg_prompt,
            style=ai_style,
            width=width,
            height=height
        )
        bg_bgr = cv2.imdecode(np.frombuffer(bg_png, np.uint8), cv2.IMREAD_COLOR)

        nodes = build_nodes(p["_resolved"], height)
        leaves = layout(nodes, width, height)
        png, placements = render(leaves, catalog, width, height, bg_bgr, shadow)
        assets = sorted({(pl["asset_image_id"], pl["asset_image_name"]) for pl in placements})
        provenance = {
            "instruction": instruction,
            "generation_mode": "hybrid",
            "plan": public,
            "placements": placements,
            "ai_elements": ai_elements,
            "ai_background": bg_meta,
            "canvas": {"width": width, "height": height, "style": ai_style},
            "assets": [{"image_id": i, "image_name": n} for i, n in assets],
            "planner": p["planner"],
            "planner_model": p.get("planner_model")
        }
        img = catalog.add_composite(png, name or _auto_name(instruction), provenance, placements, user_id=user_id)
        warnings = list(p["warnings"])
        if ai_elements:
            warnings.append(f"AI generated elements: {', '.join(ai_elements)} (not in catalog)")
        return {
            "image": img,
            "plan": public,
            "placements": placements,
            "warnings": warnings,
            "mode": "hybrid",
            "generation_mode": "hybrid"
        }

    # Mode 1: Assets Only (strict composition)
    if not p["_items"]:
        raise CompositionError("could not find any objects in the instruction", plan=public)
    if p["missing"] and not (allow_partial and p["_resolved"]):
        raise CompositionError("required catalog assets are missing - nothing was generated", p["missing"], p["suggestions"], public)
    bgname = background or p["canvas"]["background"]
    nodes = build_nodes(p["_resolved"], height)
    leaves = layout(nodes, width, height)
    png, placements = render(leaves, catalog, width, height, bgname, shadow)
    assets = sorted({(pl["asset_image_id"], pl["asset_image_name"]) for pl in placements})
    provenance = {"instruction": instruction, "generation_mode": "assets_only", "plan": public, "placements": placements,
                  "canvas": {"width": width, "height": height, "background": bgname},
                  "assets": [{"image_id": i, "image_name": n} for i, n in assets], "planner": p["planner"],
                  "planner_model": p.get("planner_model")}
    img = catalog.add_composite(png, name or _auto_name(instruction), provenance, placements, user_id=user_id)
    return {"image": img, "plan": public, "placements": placements, "warnings": p["warnings"] + (["partial: " + m for m in p["missing"]] if p["missing"] else []), "mode": "assets_only", "generation_mode": "assets_only"}


def _auto_name(instruction: str) -> str:
    s = re.sub(r"\s+", " ", instruction.strip().rstrip("."))
    return "composite: " + (s[:48] + "…" if len(s) > 48 else s)


# ----------------------------------------------------------------------------- optional LLM planner
LLM_PROMPT = """You convert an image-composition request into JSON using ONLY these catalog labels and colours.
Catalog vocabulary: {vocab}
Return JSON only: {{"canvas":{{"background":"white|<colour>|transparent"}},"items":[{{"text":str,"label":str,"count":int,"color":str|null,
"size":float(0.5-1.8, 1=normal),"anchor":"left|right|center|top|bottom|null","group":bool,"relation_to_previous":"adjacent|left_of|right_of|above|below|in_front|behind"}}]}}
group=true means "all distinct objects of that category" (e.g. 'colored shapes'). Never invent labels outside the vocabulary.
Request: {request}"""


# In compose.py
def llm_plan(instruction: str, catalog, transport=None, user_id: str | None = None):
    """Uses Gemini instead of Anthropic for optional LLM scene planning."""
    from .planner import _call_gemini_flash, _get_api_key, get_gemini_model
    key = _get_api_key()
    if transport is None and not key:
        fallback = parse_instruction(instruction)
        fallback["_llm"] = False
        return fallback
        
    vocab = sorted({norm_label(o["label"]) for o in catalog.live_objects(user_id) if o["image_kind"] == "source"} | set(CATEGORIES))
    prompt = LLM_PROMPT.format(vocab=json.dumps(vocab), request=instruction)
    
    try:
        if transport is not None:
            text = transport(prompt)
        else:
            text, actual_model = _call_gemini_flash(prompt, key, get_gemini_model())
        data = json.loads(text[text.index("{"): text.rindex("}") + 1])
        parsed = validate_llm_plan(data)
        if transport is None:
            parsed["_planner"] = f"gemini ({actual_model})"
        return parsed
    except Exception:
        return None


def validate_llm_plan(data: dict) -> dict:
    rel_ok = {"adjacent", "left_of", "right_of", "above", "below", "in_front", "behind"}
    items = []
    for i, d in enumerate(data["items"]):
        noun = SYNONYMS.get(norm_label(d["label"]), norm_label(d["label"]))
        rel = d.get("relation_to_previous", "adjacent")
        items.append(Item(f"i{i + 1}", d.get("text", noun), noun, int(max(1, min(int(d.get("count", 1)), 12))),
                          COLOR_ALIAS.get((d.get("color") or "").lower(), (d.get("color") or "").lower()) or None,
                          float(np.clip(float(d.get("size", 1.0)), 0.4, 2.0)),
                          d.get("anchor") if d.get("anchor") in ("left", "right", "center", "top", "bottom") else None,
                          bool(d.get("group", False)), rel if (rel in rel_ok and i > 0) else "adjacent"))
    if not items:
        raise ValueError("empty plan")
    bg = (data.get("canvas") or {}).get("background", "white")
    return {"canvas": {"background": bg if (bg == "transparent" or bg in BG_COLORS) else "white"}, "items": items,
            "warnings": [], "_llm": True}
