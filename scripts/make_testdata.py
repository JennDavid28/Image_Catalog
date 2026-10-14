"""Generate the assignment's test catalog (synthetic, so ground truth is exact).

Writes data/testset/<name>.png and data/testset/ground_truth.json
(label, color, bbox x,y,w,h per object).  Drawing is 3x supersampled for clean anti-aliasing.
"""
import json
from pathlib import Path
import numpy as np
from PIL import Image, ImageDraw, ImageFilter

OUT = Path(__file__).resolve().parent.parent / "data" / "testset"
SS = 3                                   # supersampling factor
rng = np.random.default_rng(7)


class Canvas:
    def __init__(self, w, h, bg=(255, 255, 255)):
        self.w, self.h = w, h
        self.img = Image.new("RGB", (w * SS, h * SS), bg)
        self.d = ImageDraw.Draw(self.img)
        self.objs = []                   # (label, color, mask 'L' image at SS scale)

    def _mask(self, fn):
        m = Image.new("L", self.img.size, 0)
        fn(ImageDraw.Draw(m))
        return m

    def shape(self, fn, fill, track=None):
        """fn(draw, fill) draws on any ImageDraw; track=(label,color) registers a GT object."""
        fn(self.d, fill)
        if track:
            m = self._mask(lambda dr: fn(dr, 255))
            self.objs.append((track[0], track[1], m))

    def track_extra(self, idx, fn):
        """add more pixels (stem, leaf...) to an already tracked object's GT mask"""
        fn(ImageDraw.Draw(self.objs[idx][2]), 255)

    def shaded_ellipse(self, box, dark, light, hl=None):
        """ellipse with radial gradient (cartoon-realistic shading) + optional highlight"""
        x0, y0, x1, y1 = [v * SS for v in box]
        m = self._mask(lambda dr: dr.ellipse((x0, y0, x1, y1), fill=255))
        yy, xx = np.mgrid[0:self.img.size[1], 0:self.img.size[0]]
        lx, ly = x0 + (x1 - x0) * 0.35, y0 + (y1 - y0) * 0.3
        r = np.hypot(xx - lx, yy - ly) / (0.9 * max(x1 - x0, y1 - y0))
        t = np.clip(r, 0, 1)[..., None]
        grad = (np.array(light) * (1 - t) + np.array(dark) * t).astype(np.uint8)
        self.img.paste(Image.fromarray(grad), (0, 0), m)
        if hl:
            hx, hy = lx, ly
            hm = self._mask(lambda dr: dr.ellipse((hx - 30 * SS, hy - 18 * SS, hx + 10 * SS, hy + 10 * SS), fill=170))
            self.img.paste(Image.new("RGB", self.img.size, hl), (0, 0), hm.filter(ImageFilter.GaussianBlur(6)))
        return m

    def save(self, name, noise=2.0):
        im = self.img.resize((self.w, self.h), Image.LANCZOS)
        a = np.asarray(im).astype(np.float32)
        a += rng.normal(0, noise, a.shape)
        Image.fromarray(np.clip(a, 0, 255).astype(np.uint8)).save(OUT / f"{name}.png")
        gt = []
        for label, color, m in self.objs:
            m = np.asarray(m.resize((self.w, self.h), Image.LANCZOS)) > 127
            ys, xs = np.where(m)
            gt.append({"label": label, "color": color, "bbox": [int(xs.min()), int(ys.min()),
                                                               int(xs.max() - xs.min() + 1), int(ys.max() - ys.min() + 1)]})
        return gt


def S(v):  # scale helper
    return [int(x * SS) for x in v]


def apple(c, cx, cy, r, track=True):
    """red cartoon-realistic apple: 2 lobes + shading + stem + leaf"""
    lobe = lambda dr, f: (dr.ellipse(S((cx - r, cy - r * .82, cx + r * .12, cy + r * .95)), fill=f),
                          dr.ellipse(S((cx - r * .12, cy - r * .82, cx + r, cy + r * .95)), fill=f))
    m = c._mask(lambda dr: lobe(dr, 255))
    yy, xx = np.mgrid[0:c.img.size[1], 0:c.img.size[0]]
    lx, ly = (cx - r * .35) * SS, (cy - r * .3) * SS
    t = np.clip(np.hypot(xx - lx, yy - ly) / (1.7 * r * SS), 0, 1)[..., None]
    grad = (np.array((235, 60, 60)) * (1 - t) + np.array((140, 10, 25)) * t).astype(np.uint8)
    c.img.paste(Image.fromarray(grad), (0, 0), m)
    hm = c._mask(lambda dr: dr.ellipse(S((cx - r * .62, cy - r * .55, cx - r * .3, cy - r * .3)), fill=190))
    c.img.paste(Image.new("RGB", c.img.size, (255, 225, 225)), (0, 0), hm.filter(ImageFilter.GaussianBlur(5)))
    stem = lambda dr, f: dr.line(S((cx, cy - r * .7, cx + r * .08, cy - r * 1.1)), fill=f, width=int(9 * SS * r / 150))
    leaf = lambda dr, f: dr.ellipse(S((cx + r * .1, cy - r * 1.12, cx + r * .5, cy - r * .92)), fill=f)
    stem(c.d, (90, 55, 30)); leaf(c.d, (50, 150, 60))
    if track:
        m2 = c._mask(lambda dr: (lobe(dr, 255), stem(dr, 255), leaf(dr, 255)))
        c.objs.append(("apple", "red", m2))


def orange(c, cx, cy, r):
    m = c.shaded_ellipse((cx - r, cy - r, cx + r, cy + r), dark=(215, 105, 5), light=(255, 170, 40))
    # dimple texture
    dots = c._mask(lambda dr: [dr.ellipse(S((x, y, x + 2.2, y + 2.2)), fill=90) for x, y in
                               zip(rng.uniform(cx - r, cx + r, 260), rng.uniform(cy - r, cy + r, 260))])
    from PIL import ImageChops
    dots = ImageChops.multiply(dots, m)
    c.img.paste(Image.new("RGB", c.img.size, (200, 90, 0)), (0, 0), dots)
    c.objs.append(("orange", "orange", m))


def person(c, cx, kind, top, bottom, hair, skin=(255, 205, 160), y0=60):
    """cartoon figure ~ 700px tall: head, hair, shirt, arms, legs/skirt, shoes. GT = whole figure"""
    parts = []
    def add(fn, fill): c.shape(fn, fill); parts.append(fn)
    # legs / skirt
    if kind == "girl":
        add(lambda dr, f: dr.rectangle(S((cx - 38, y0 + 420, cx - 14, y0 + 600)), fill=f), skin)
        add(lambda dr, f: dr.rectangle(S((cx + 14, y0 + 420, cx + 38, y0 + 600)), fill=f), skin)
        add(lambda dr, f: dr.polygon(S((cx - 62, y0 + 300, cx + 62, y0 + 300, cx + 120, y0 + 450, cx - 120, y0 + 450)), fill=f), bottom)
        add(lambda dr, f: dr.rounded_rectangle(S((cx - 52, y0 + 600, cx - 8, y0 + 625)), 6 * SS, fill=f), (200, 40, 50))
        add(lambda dr, f: dr.rounded_rectangle(S((cx + 8, y0 + 600, cx + 52, y0 + 625)), 6 * SS, fill=f), (200, 40, 50))
    else:
        add(lambda dr, f: dr.rectangle(S((cx - 62, y0 + 330, cx - 4, y0 + 590)), fill=f), bottom)
        add(lambda dr, f: dr.rectangle(S((cx + 4, y0 + 330, cx + 62, y0 + 590)), fill=f), bottom)
        add(lambda dr, f: dr.rounded_rectangle(S((cx - 68, y0 + 590, cx - 2, y0 + 618)), 8 * SS, fill=f), (30, 30, 30))
        add(lambda dr, f: dr.rounded_rectangle(S((cx + 2, y0 + 590, cx + 68, y0 + 618)), 8 * SS, fill=f), (30, 30, 30))
    # torso + arms
    add(lambda dr, f: dr.rounded_rectangle(S((cx - 64, y0 + 190, cx + 64, y0 + 340)), 14 * SS, fill=f), top)
    add(lambda dr, f: dr.rounded_rectangle(S((cx - 100, y0 + 200, cx - 62, y0 + 330)), 14 * SS, fill=f), top)
    add(lambda dr, f: dr.rounded_rectangle(S((cx + 62, y0 + 200, cx + 100, y0 + 330)), 14 * SS, fill=f), top)
    add(lambda dr, f: dr.ellipse(S((cx - 104, y0 + 326, cx - 58, y0 + 366)), fill=f), skin)
    add(lambda dr, f: dr.ellipse(S((cx + 58, y0 + 326, cx + 104, y0 + 366)), fill=f), skin)
    # neck, head, hair, face
    add(lambda dr, f: dr.rectangle(S((cx - 18, y0 + 150, cx + 18, y0 + 200)), fill=f), skin)
    add(lambda dr, f: dr.ellipse(S((cx - 72, y0, cx + 72, y0 + 160)), fill=f), skin)
    if kind == "girl":
        add(lambda dr, f: dr.pieslice(S((cx - 78, y0 - 8, cx + 78, y0 + 150)), 180, 360, fill=f), hair)
        add(lambda dr, f: dr.ellipse(S((cx - 112, y0 + 40, cx - 66, y0 + 100)), fill=f), hair)
        add(lambda dr, f: dr.ellipse(S((cx + 66, y0 + 40, cx + 112, y0 + 100)), fill=f), hair)
    else:
        add(lambda dr, f: dr.pieslice(S((cx - 76, y0 - 10, cx + 76, y0 + 120)), 180, 360, fill=f), hair)
    for ex in (-26, 26):
        c.d.ellipse(S((cx + ex - 7, y0 + 70, cx + ex + 7, y0 + 86)), fill=(30, 30, 30))
    c.d.arc(S((cx - 30, y0 + 90, cx + 30, y0 + 130)), 20, 160, fill=(150, 50, 50), width=3 * SS)
    # one GT object for the whole figure
    full = c._mask(lambda dr: [p(dr, 255) for p in parts])
    c.objs.append((kind, f"{top_name[kind]}/{bottom_name[kind]}", full))


top_name = {"boy": "green", "girl": "purple"}
bottom_name = {"boy": "blue", "girl": "pink"}


def build():
    OUT.mkdir(parents=True, exist_ok=True)
    gt = {}

    c = Canvas(512, 512); apple(c, 256, 300, 150); gt["apple"] = c.save("apple")
    c = Canvas(512, 512); orange(c, 256, 270, 150); gt["orange"] = c.save("orange")

    c = Canvas(512, 800)
    person(c, 256, "boy", top=(50, 170, 70), bottom=(50, 70, 190), hair=(100, 60, 30)); gt["boy"] = c.save("boy")
    c = Canvas(512, 800)
    person(c, 256, "girl", top=(135, 60, 175), bottom=(255, 120, 175), hair=(250, 205, 60)); gt["girl"] = c.save("girl")

    c = Canvas(640, 640)
    c.shape(lambda dr, f: dr.ellipse(S((85, 85, 275, 275)), fill=f), (225, 40, 40), ("circle", "red"))
    c.shape(lambda dr, f: dr.rectangle(S((370, 90, 560, 280)), fill=f), (40, 90, 225), ("square", "blue"))
    c.shape(lambda dr, f: dr.polygon(S((100, 560, 290, 560, 195, 380)), fill=f), (45, 170, 70), ("triangle", "green"))
    c.shape(lambda dr, f: dr.rectangle(S((350, 400, 580, 540)), fill=f), (250, 215, 40), ("rectangle", "yellow"))
    gt["shapes"] = c.save("shapes")

    c = Canvas(800, 300)
    for i, (col, name) in enumerate([((225, 40, 40), "red"), ((40, 90, 225), "blue"), ((45, 170, 70), "green"),
                                     ((250, 215, 40), "yellow"), ((140, 60, 180), "purple")]):
        c.shape(lambda dr, f, i=i: dr.rectangle(S((30 + i * 148, 60, 30 + (i + 1) * 148 - 1, 240)), fill=f), col, ("color swatch", name))
    gt["colors"] = c.save("colors")

    c = Canvas(900, 512)
    apple(c, 240, 290, 140); orange(c, 650, 280, 125)
    c.objs = []
    # re-track with exact masks
    c2 = Canvas(900, 512); apple(c2, 240, 290, 140); o1 = c2.objs[:]; orange(c2, 650, 280, 125)
    gt["apple_orange"] = c2.save("apple_orange")

    c = Canvas(640, 640)
    c.shape(lambda dr, f: dr.ellipse(S((70, 120, 330, 380)), fill=f), (225, 40, 40), ("circle", "red"))
    c.shape(lambda dr, f: dr.polygon(S((290, 560, 520, 560, 405, 340)), fill=f), (45, 170, 70), ("triangle", "green"))
    c.shape(lambda dr, f: dr.rectangle(S((270, 200, 480, 410)), fill=f), (40, 90, 225), ("square", "blue"))
    gt["overlapping_shapes"] = c.save("overlapping_shapes")

    c = Canvas(400, 400); gt["blank"] = c.save("blank")                       # nothing to detect
    c = Canvas(400, 400)                                                       # object too faint for any detector
    c.shape(lambda dr, f: dr.ellipse(S((100, 100, 300, 300)), fill=f), (240, 240, 240), ("circle", "light gray"))
    gt["low_contrast"] = c.save("low_contrast", noise=1.0)

    (OUT / "ground_truth.json").write_text(json.dumps(gt, indent=2))
    return gt


if __name__ == "__main__":
    g = build()
    for k, v in g.items():
        print(f"{k:20s}", [(o['label'], o['color'], o['bbox']) for o in v])
