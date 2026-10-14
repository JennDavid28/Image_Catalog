"""Metadata + manual-correction rules: manual edits are the source of truth until detection is explicitly re-run."""
import pytest

from app.catalog import Catalog, CatalogError, Duplicate, NotFound
from tests.conftest import img_bytes, by_name


def test_auto_metadata_is_stored(cat):
    img = cat.add_image(img_bytes("apple"), "apple", "apple.png")
    a = img["auto"]
    assert (a["width"], a["height"], a["format"]) == (512, 512, "png")
    assert a["detection_status"] == "ok" and a["detector_version"] and a["sha256"]
    o = img["objects"][0]
    assert o["label"] == "apple" and o["color"] == "red" and len(o["bbox"]) == 4 and 0 < o["confidence"] <= 1
    assert set(o["field_source"].values()) == {"auto"} and o["origin"] == "auto"
    assert img["user"]["display_name"] == "apple"          # user-editable block is separate from the auto block


def test_duplicate_upload_rejected(cat):
    cat.add_image(img_bytes("apple"), "apple")
    with pytest.raises(Duplicate):
        cat.add_image(img_bytes("apple"), "again")


def test_non_image_rejected(cat):
    with pytest.raises(CatalogError):
        cat.add_image(b"not an image", "x")


def test_failed_detection_then_manual_add(cat):
    img = cat.add_image(img_bytes("low_contrast"), "lc")
    assert img["auto"]["detection_status"] == "no_objects" and img["objects"] == []
    o = cat.add_object(img["id"], "circle", [100, 100, 200, 200], color="light gray", confidence=0.9)
    assert o["origin"] == "manual" and o["reviewed"]
    assert o["field_source"] == {"label": "user", "confidence": "user", "bbox": "user", "color": "user"}
    assert o["mask_source"] in ("matte", "grabcut", "box")


def test_manual_add_infers_colour_but_marks_it_auto(cat):
    img = cat.add_image(img_bytes("low_contrast"), "lc")
    o = cat.add_object(img["id"], "circle", [100, 100, 200, 200])
    assert o["color"] == "light gray" and o["field_source"]["color"] == "auto" and o["field_source"]["label"] == "user"


def test_edit_marks_only_changed_fields_and_keeps_detected_values(seeded):
    img = by_name(seeded, "shapes"); circle = next(o for o in seeded.get_image(img["id"])["objects"] if o["label"] == "circle")
    e = seeded.update_object(circle["id"], {"label": "red ball", "confidence": 0.5})
    assert e["field_source"]["label"] == "user" and e["field_source"]["confidence"] == "user"
    assert e["field_source"]["color"] == "auto" and e["field_source"]["bbox"] == "auto"
    assert e["auto_values"]["label"] == "circle"                      # original detection is never lost
    assert e["edited"] and not e["needs_review"]


def test_edit_same_value_is_not_an_edit(seeded):
    img = by_name(seeded, "apple"); o = seeded.get_image(img["id"])["objects"][0]
    e = seeded.update_object(o["id"], {"label": o["label"], "bbox": o["bbox"]})
    assert not e["edited"]


def test_revert_restores_detected_values(seeded):
    img = by_name(seeded, "apple"); o = seeded.get_image(img["id"])["objects"][0]
    seeded.update_object(o["id"], {"label": "pear", "bbox": [0, 0, 50, 50]})
    r = seeded.revert_object(o["id"])
    assert r["label"] == "apple" and r["bbox"] == o["bbox"] and not r["edited"]


def test_edits_persist_across_restart(tmp_path):
    c1 = Catalog(tmp_path / "s"); img = c1.add_image(img_bytes("shapes"), "shapes")
    tri = next(o for o in img["objects"] if o["label"] == "triangle")
    c1.update_object(tri["id"], {"label": "green triangle", "confidence": 0.7})
    c1.add_object(img["id"], "sticker", [10, 10, 40, 40])
    c2 = Catalog(tmp_path / "s")                                       # brand new process / instance
    objs = {o["label"]: o for o in c2.get_image(img["id"])["objects"]}
    assert "green triangle" in objs and objs["green triangle"]["confidence"] == 0.7 and "sticker" in objs


def _setup_corrections(cat):
    img = cat.add_image(img_bytes("shapes"), "shapes")
    objs = {o["label"]: o for o in img["objects"]}
    cat.update_object(objs["triangle"]["id"], {"label": "green triangle"})             # edited
    cat.delete_object(objs["circle"]["id"])                                             # removed
    cat.confirm_object(objs["square"]["id"])                                            # reviewed as-is
    cat.add_object(img["id"], "sticker", [5, 5, 30, 30])                                # added
    return img["id"]


def test_normal_operations_never_rerun_detection(cat):
    iid = _setup_corrections(cat)
    before = cat.get_image(iid)["auto"]["detected_at"]
    cat.update_image(iid, display_name="renamed", tags=["a"]); cat.list_images(); cat.get_image(iid)
    assert cat.get_image(iid)["auto"]["detected_at"] == before
    runs = [e for e in cat.events(iid) if e["action"] == "detect"]
    assert [r["detail"]["mode"] for r in runs] == ["initial"]


def test_rerun_preserve_manual_keeps_all_corrections(cat):
    iid = _setup_corrections(cat)
    out = cat.detect(iid, "preserve_manual")
    live = {o["label"]: o for o in out["objects"] if not o["deleted"]}
    assert "green triangle" in live and "sticker" in live and "square" in live and "rectangle" in live
    assert "circle" not in live                                       # removed object stays removed (tombstone)
    assert len([o for o in out["objects"] if o["label"] == "circle"]) == 1 and next(o for o in out["objects"] if o["label"] == "circle")["deleted"]
    assert len(live) == 4                                             # nothing duplicated


def test_rerun_replace_all_discards_manual_work_only_on_explicit_request(cat):
    iid = _setup_corrections(cat)
    out = cat.detect(iid, "replace_all")
    labels = sorted(o["label"] for o in out["objects"])
    assert labels == ["circle", "rectangle", "square", "triangle"] and all(o["origin"] == "auto" and not o["edited"] for o in out["objects"])


def test_rerun_requires_explicit_mode(cat):
    iid = cat.add_image(img_bytes("apple"), "a")["id"]
    with pytest.raises(CatalogError):
        cat.detect(iid, "whatever")


def test_changing_box_changes_the_cutout_asset(seeded):
    img = by_name(seeded, "apple"); o = seeded.get_image(img["id"])["objects"][0]
    a0 = seeded.object_asset(o["id"])
    x, y, w, h = o["bbox"]
    seeded.update_object(o["id"], {"bbox": [x, y, w // 2, h]})
    a1 = seeded.object_asset(o["id"])
    assert a1.shape[1] < a0.shape[1] and a1.shape[2] == 4


def test_validation_errors(seeded):
    img = by_name(seeded, "apple"); o = seeded.get_image(img["id"])["objects"][0]
    with pytest.raises(CatalogError): seeded.update_object(o["id"], {"confidence": 1.5})
    with pytest.raises(CatalogError): seeded.update_object(o["id"], {"label": "  "})
    with pytest.raises(CatalogError): seeded.update_object(o["id"], {"bbox": [1, 2, 3]})
    with pytest.raises(CatalogError): seeded.add_object(img["id"], "", [0, 0, 10, 10])
    with pytest.raises(NotFound): seeded.update_object("nope", {"label": "x"})
    big = seeded.update_object(o["id"], {"bbox": [-50, -50, 9999, 9999]})          # clamped, never out of the image
    assert big["bbox"][0] >= 0 and big["bbox"][0] + big["bbox"][2] <= 512


def test_audit_log_records_every_change(cat):
    iid = _setup_corrections(cat)
    actions = [e["action"] for e in cat.events(iid)]
    for a in ("image_added", "detect", "object_edit", "object_deleted", "object_confirmed", "object_added"):
        assert a in actions


def test_delete_image_cleans_up(tmp_path):
    c = Catalog(tmp_path / "s"); img = c.add_image(img_bytes("apple"), "a")
    path = c.image_path(img["id"]); assert path.exists()
    c.delete_image(img["id"])
    assert not path.exists() and c.list_images() == []
    with pytest.raises(NotFound): c.get_image(img["id"])
