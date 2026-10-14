"""Trash folder, class colours, mask outlines, image editing, detector tiers and the open-vocabulary hook."""
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app import detector, imageops, openvocab
from app.api import create_app
from app.catalog import CatalogError
from tests.conftest import img_bytes


@pytest.fixture()
def client(tmp_path):
    client = TestClient(create_app(tmp_path / "store"))
    client.post("/api/auth/register", json={"username": "tester", "display_name": "Test User", "password": "testpass123"})
    client.headers["Authorization"] = "Bearer " + client.post("/api/auth/login", json={"username": "tester", "password": "testpass123"}).json()["token"]
    return client


def up(client, name):
    return client.post("/api/images", files={"file": (f"{name}.png", img_bytes(name), "image/png")}).json()


# ----------------------------------------------------------------------------- trash
def test_trash_restore_and_purge(client):
    a, b = up(client, "apple"), up(client, "orange")
    assert client.delete(f"/api/images/{a['id']}").json()["trashed"] is True
    assert [i["id"] for i in client.get("/api/images").json()] == [b["id"]]          # gone from the catalog
    t = client.get("/api/trash").json()
    assert [i["id"] for i in t["images"]] == [a["id"]]
    assert client.get(f"/api/images/{a['id']}/file").status_code == 200              # files are kept while in trash
    assert client.delete(f"/api/trash/{b['id']}").status_code == 400                 # cannot purge what is not trashed
    assert client.post(f"/api/images/{a['id']}/restore").json()["auto"]["trashed_at"] is None
    assert len(client.get("/api/images").json()) == 2
    client.delete(f"/api/images/{a['id']}")
    assert client.delete(f"/api/trash/{a['id']}").json()["purged"] == a["id"]
    assert client.get(f"/api/images/{a['id']}").status_code == 404 and client.get("/api/trash").json()["images"] == []


def test_empty_trash_and_trashed_images_are_not_composer_assets(client):
    a, b = up(client, "apple"), up(client, "orange")
    client.delete(f"/api/images/{a['id']}")
    r = client.post("/api/compose/plan", json={"instruction": "one red apple"}).json()
    assert r["missing"], "a trashed image must not be usable as an asset"
    client.post(f"/api/images/{a['id']}/restore"); client.delete(f"/api/images/{a['id']}"); client.delete(f"/api/images/{b['id']}")
    assert client.delete("/api/trash").json()["purged"] == 2 and client.get("/api/trash").json()["images"] == []


def test_removed_objects_listed_in_trash(client):
    d = up(client, "shapes"); o = d["objects"][0]
    client.delete(f"/api/objects/{o['id']}")
    t = client.get("/api/trash").json()
    assert [x["id"] for x in t["objects"]] == [o["id"]] and t["objects"][0]["image_name"] == "shapes"
    client.post(f"/api/objects/{o['id']}/restore")
    assert client.get("/api/trash").json()["objects"] == []


def test_old_catalog_is_migrated(tmp_path):
    import sqlite3
    from app.catalog import Catalog
    store = tmp_path / "old"; Catalog(store)
    con = sqlite3.connect(store / "catalog.db"); con.execute("ALTER TABLE images DROP COLUMN trashed_at"); con.commit(); con.close()
    c = Catalog(store); assert c.list_trash() == {"images": [], "objects": []}


# ----------------------------------------------------------------------------- classes + polygons
def test_class_colours_are_stable_distinct_and_editable(client):
    up(client, "apple_orange")
    cl = client.get("/api/classes").json()
    assert {c["label"] for c in cl} == {"apple", "orange"} and len({c["color"] for c in cl}) == 2
    assert client.get("/api/classes").json() == cl                                   # stable across calls
    r = client.patch("/api/classes/apple", json={"color": "#FF0000"}); assert r.json()["color"] == "#ff0000"
    assert next(c for c in client.get("/api/classes").json() if c["label"] == "apple")["color"] == "#ff0000"
    assert client.patch("/api/classes/apple", json={"color": "nope"}).status_code == 422


def test_polygons_follow_the_object_shape(client):
    d = up(client, "shapes"); polys = client.get(f"/api/images/{d['id']}/polygons").json()
    assert set(polys) == {o["id"] for o in d["objects"]}
    circle = next(o for o in d["objects"] if o["label"] == "circle")
    p = polys[circle["id"]]; assert p["color"].startswith("#") and p["polys"] and len(p["polys"][0]) >= 8   # a round outline, not a 4-point box


# ----------------------------------------------------------------------------- image ops (pure)
def _flat():
    im = np.full((80, 120, 3), 255, np.uint8); im[20:60, 30:90] = (40, 40, 200); return im


def test_crop_resize_rotate_flip():
    im = _flat()
    assert imageops.crop(im, 10, 10, 50, 40).shape[:2] == (40, 50)
    assert imageops.crop(im, 100, 60, 500, 500).shape[:2] == (20, 20)                 # clamps to the image
    with pytest.raises(imageops.EditError): imageops.crop(im, 0, 0, 2, 2)
    assert imageops.resize(im, percent=50).shape[:2] == (40, 60)
    assert imageops.resize(im, width=60).shape[:2] == (40, 60)                        # keeps aspect when one side is given
    with pytest.raises(imageops.EditError): imageops.resize(im, width=99999)
    assert imageops.rotate(im, 90).shape[:2] == (120, 80) and imageops.rotate(im, 180).shape == im.shape
    with pytest.raises(imageops.EditError): imageops.rotate(im, 45)
    assert (imageops.flip(im, "horizontal")[40, 119 - 30] == im[40, 30]).all()


def test_adjust_matches_css_definitions():
    im = np.full((4, 4, 3), (100, 150, 200), np.uint8)
    assert (imageops.adjust(im) == im).all()                                          # defaults are a no-op
    assert imageops.adjust(im, brightness=50)[0, 0].tolist() == [50, 75, 100]
    g = imageops.adjust(im, grayscale=True)[0, 0]; assert g[0] == g[1] == g[2]
    assert imageops.adjust(im, saturation=0)[0, 0].std() < 1.5
    with pytest.raises(imageops.EditError): imageops.adjust(im, brightness=900)


def test_recolor_only_touches_the_mask_and_keeps_shading():
    im = np.full((60, 60, 3), 255, np.uint8); im[10:50, 10:50] = (30, 30, 200)
    im[10:30, 10:50] = (60, 60, 250)                                                  # lighter top half = shading
    m = np.zeros((60, 60), np.uint8); m[10:50, 10:50] = 255
    out = imageops.recolor_region(im, m, "#00aa00")
    assert (out[2, 2] == 255).all()                                                   # background untouched
    top, bot = out[20, 30].astype(int), out[40, 30].astype(int)
    assert top[1] > top[2] and bot[1] > bot[2]                                        # both halves are green now
    assert top.max() > bot.max()                                                      # ...and the highlight is still lighter
    with pytest.raises(imageops.EditError): imageops.recolor_region(im, m, "green")


def test_cutout_is_transparent_outside_and_erase_fills():
    im = _flat(); m = np.zeros(im.shape[:2], np.uint8); m[20:60, 30:90] = 255
    cut = imageops.cutout(im, m); assert cut.shape[2] == 4 and cut.shape[:2] == (40, 60) and cut[20, 30, 3] > 200
    assert cut[0, 0, 3] < 200                                                         # eroded, feathered edge
    er = imageops.erase(im, m); assert er[40, 60].tolist() == [255, 255, 255]         # object removed, backdrop restored
    with pytest.raises(imageops.EditError): imageops.erase(im, np.ones(im.shape[:2], np.uint8) * 255)


# ----------------------------------------------------------------------------- image ops (through the API)
def test_edit_copy_keeps_original_and_redetects(client):
    d = up(client, "shapes"); W, H = d["auto"]["width"], d["auto"]["height"]
    r = client.post(f"/api/images/{d['id']}/edit", json={"op": "crop", "params": {"x": 0, "y": 0, "width": W // 2, "height": H}})
    assert r.status_code == 201; new = r.json()["image"]
    assert new["id"] != d["id"] and new["auto"]["width"] == W // 2 and new["user"]["display_name"].endswith("(cropped)")
    assert new["auto"]["provenance"]["edited_from"] == d["id"] and new["auto"]["detection_status"] != "pending"
    assert client.get(f"/api/images/{d['id']}").json()["auto"]["width"] == W                      # original untouched
    assert len(client.get("/api/images").json()) == 2


def test_edit_replace_overwrites_and_redetects(client):
    d = up(client, "apple"); W = d["auto"]["width"]
    r = client.post(f"/api/images/{d['id']}/edit", json={"op": "rotate", "params": {"degrees": 90}, "mode": "replace"}).json()["image"]
    assert r["id"] == d["id"] and r["auto"]["height"] == W and len(client.get("/api/images").json()) == 1


def test_object_cut_recolor_erase(client):
    d = up(client, "apple_orange"); apple = next(o for o in d["objects"] if o["label"] == "apple")
    cut = client.post(f"/api/images/{d['id']}/edit", json={"op": "cut_object", "params": {"object_id": apple["id"]}}).json()["image"]
    assert cut["auto"]["format"] == "png" and cut["user"]["display_name"].endswith("(cut-out)")
    raw = cv2.imdecode(np.frombuffer(client.get(f"/api/images/{cut['id']}/file").content, np.uint8), cv2.IMREAD_UNCHANGED)
    assert raw.shape[2] == 4 and (raw[..., 3] == 0).any() and (raw[..., 3] == 255).any()           # real transparency
    rec = client.post(f"/api/images/{d['id']}/edit", json={"op": "recolor_object", "params": {"object_id": apple["id"], "color": "#2266ff"}}).json()["image"]
    px = cv2.imdecode(np.frombuffer(client.get(f"/api/images/{rec['id']}/file").content, np.uint8), cv2.IMREAD_COLOR)
    x, y, w, h = apple["bbox"]; b, g, r_ = px[y + h // 2, x + w // 2].astype(int)
    assert b > r_ + 40, "the apple's pixels should now be blue-dominant"
    ox, oy, ow, oh = next(o for o in d["objects"] if o["label"] == "orange")["bbox"]
    before = cv2.imdecode(np.frombuffer(client.get(f"/api/images/{d['id']}/file").content, np.uint8), cv2.IMREAD_COLOR)
    assert (px[oy + oh // 2, ox + ow // 2] == before[oy + oh // 2, ox + ow // 2]).all()             # the orange is untouched
    er = client.post(f"/api/images/{d['id']}/edit", json={"op": "erase_object", "params": {"object_id": apple["id"]}}).json()["image"]
    assert not any(o["label"] == "apple" and not o["deleted"] for o in er["objects"])               # the apple is really gone


def test_edit_validation(client):
    d = up(client, "apple")
    assert client.post(f"/api/images/{d['id']}/edit", json={"op": "sparkle"}).status_code == 400
    assert client.post(f"/api/images/{d['id']}/edit", json={"op": "resize", "params": {}}).status_code == 400
    assert client.post(f"/api/images/{d['id']}/edit", json={"op": "recolor_object", "params": {"color": "#fff"}}).status_code == 400
    other = up(client, "orange")["objects"][0]["id"]
    assert client.post(f"/api/images/{d['id']}/edit", json={"op": "cut_object", "params": {"object_id": other}}).status_code == 400


# ----------------------------------------------------------------------------- detector tiers + open vocabulary
def test_capabilities_report_is_honest(client):
    c = client.get("/api/capabilities").json()
    assert c["coco_model"] is True and isinstance(c["open_vocabulary"], bool)
    if not c["open_vocabulary"]:
        assert "COCO" in c["known_classes"] and c["open_vocabulary_note"]


def test_weak_photo_proposals_are_flagged_for_review():
    im = cv2.imread(str(__import__("pathlib").Path(__file__).parent / "real" / "orange.jpg"))
    r = detector.detect(im)
    assert r.objects
    for d in r.objects:
        if d.attrs.get("weak"):
            assert d.confidence < detector.NEEDS_REVIEW


def test_open_vocab_results_merge_with_yolox(monkeypatch):
    """A fake open-vocabulary backend adds 'pencil' (unknown to COCO) and must flow through the normal pipeline."""
    im = cv2.imread(str(__import__("pathlib").Path(__file__).parent / "real" / "orange.jpg"))
    h, w = im.shape[:2]
    class Fake:
        def available(self): return True
        def status(self): return {"available": True, "reason": "fake"}
        def detect(self, bgr, extra=None, **k): return [{"label": "pencil", "score": 0.6, "bbox": (5, 5, 40, 90)}] + \
            [{"label": e, "score": 0.5, "bbox": (w - 60, h - 60, 50, 50)} for e in (extra or [])]
    monkeypatch.setattr(detector, "OPENVOCAB", Fake())
    r = detector.detect(im, extra_labels=["gizmo"])
    labels = {d.label for d in r.objects}
    assert "pencil" in labels and "gizmo" in labels
    pen = next(d for d in r.objects if d.label == "pencil"); assert pen.attrs["source"] == "open-vocab" and "open-vocabulary" in pen.evidence[0]
    assert r.route.endswith("open-vocabulary")


def test_yolo_world_gets_generic_vocabulary_on_normal_detection(monkeypatch):
    from app.yoloworld import YOLOWORLD
    captured = {}
    monkeypatch.setattr(YOLOWORLD, "available", lambda: True)
    def fake_world_detect(bgr, classes, **kwargs):
        captured["classes"] = classes
        return [], 1.0
    monkeypatch.setattr(YOLOWORLD, "detect", fake_world_detect)
    monkeypatch.setattr(detector.OPENVOCAB, "available", lambda: False)

    class NoYoloX:
        def available(self): return False

    _, _, world_used = detector._proposals(np.zeros((32, 32, 3), np.uint8), NoYoloX(), 0.15, ["gizmo"])
    assert world_used is True
    assert {"laptop", "cup", "remote", "gizmo"} <= set(captured["classes"])


def test_openvocab_degrades_quietly_without_torch():
    ov = openvocab.OpenVocabDetector()
    if not ov.available():
        assert ov.detect(np.zeros((50, 50, 3), np.uint8)) == []
    v = openvocab._vocab_list(["gizmo", "Pencil"]); assert "gizmo" in v and v.count("pencil") == 1 and "crumpled paper" in v
