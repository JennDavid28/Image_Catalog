"""Composite generation: real catalog pixels only, never silent approximations."""
import cv2
import numpy as np
import pytest

from app import compose
from app.compose import CompositionError, parse_instruction
from app.detector import iou
from tests.conftest import img_bytes, by_name


def _items(text):
    return [(i.noun, i.count, i.color, i.group) for i in parse_instruction(text)["items"]]


def test_parser_assignment_examples():
    assert _items("Create an image containing one red apple and one orange") == [("apple", 1, "red", False), ("orange", 1, None, False)]
    assert _items("Generate a composite containing a boy, a girl, and colored shapes") == [
        ("boy", 1, None, False), ("girl", 1, None, False), ("shape", 1, None, True)]


def test_parser_counts_sizes_relations_background():
    p = parse_instruction("a small apple to the left of a large orange on a blue background")
    assert p["canvas"]["background"] == "blue"
    assert [i.size for i in p["items"]] == [0.7, 1.35] and p["items"][1].relation == "left_of"
    assert _items("two apples")[0][1] == 2 and _items("3 oranges")[0][1] == 3
    assert parse_instruction("an orange above a red apple on a transparent background")["canvas"]["background"] == "transparent"
    assert [i.anchor for i in parse_instruction("a boy on the left, a girl on the right")["items"]] == ["left", "right"]


def test_apple_and_orange_composite(seeded):
    n_before = len(seeded.list_images())
    r = compose.compose(seeded, "Create an image containing one red apple and one orange")
    img = r["image"]
    assert len(seeded.list_images()) == n_before + 1                                   # saved as a NEW catalog entry
    assert img["kind"] == "composite" and img["id"] not in {i["id"] for i in seeded.list_images()[:n_before]}
    assert (img["auto"]["width"], img["auto"]["height"]) == (1024, 768)
    labels = sorted((o["label"], o["color"]) for o in img["objects"])
    assert labels == [("apple", "red"), ("orange", "orange")]
    assert img["auto"]["provenance"]["instruction"].startswith("Create an image")
    assert {a["image_name"] for a in img["auto"]["provenance"]["assets"]} <= {"apple", "orange", "apple_orange"}
    assert all(o["origin"] == "composite" and o["confidence"] == 1.0 for o in img["objects"])
    boxes = [o["bbox"] for o in img["objects"]]
    assert iou(boxes[0], boxes[1]) == 0.0                                              # side by side, no accidental overlap
    for x, y, w, h in boxes:
        assert x >= 0 and y >= 0 and x + w <= 1024 and y + h <= 768


def test_composite_uses_the_real_catalog_pixels(seeded):
    """Fidelity: pasted region must match the source object's pixels (not a regenerated lookalike)."""
    r = compose.compose(seeded, "one orange", shadow=False)
    pl = r["placements"][0]
    comp = cv2.imdecode(np.frombuffer(seeded.image_path(r["image"]["id"]).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
    asset = seeded.object_asset(pl["asset_object_id"])
    x, y, w, h = pl["bbox"]
    resized = cv2.resize(asset, (w, h), interpolation=cv2.INTER_AREA)
    alpha = resized[..., 3] > 250
    diff = np.abs(comp[y:y + h, x:x + w].astype(int) - resized[..., :3].astype(int)).mean(axis=2)
    assert alpha.sum() > 1000 and diff[alpha].mean() < 3.0


def test_boy_girl_and_shapes(seeded):
    r = compose.compose(seeded, "Generate a composite containing a boy, a girl, and colored shapes")
    got = [o["label"] for o in r["image"]["objects"]]
    assert got == ["boy", "girl", "circle", "square", "triangle", "rectangle"] or sorted(got) == sorted(["boy", "girl", "circle", "square", "triangle", "rectangle"])
    assert {o["attrs"]["asset_image_name"] for o in r["image"]["objects"]} == {"boy", "girl", "shapes"}
    boxes = [o["bbox"] for o in r["image"]["objects"]]
    assert all(iou(boxes[i], boxes[j]) < 0.05 for i in range(6) for j in range(i + 1, 6))


def test_missing_asset_fails_loudly_and_creates_nothing(seeded):
    n = len(seeded.list_images())
    with pytest.raises(CompositionError) as e:
        compose.compose(seeded, "two apples and a banana")
    assert any("banana" in m for m in e.value.missing) and len(seeded.list_images()) == n


def test_missing_asset_suggests_close_labels(seeded):
    with pytest.raises(CompositionError) as e:
        compose.compose(seeded, "an aple")
    assert "apple" in sum(e.value.suggestions.values(), [])


def test_colour_constraint_is_never_silently_relaxed(seeded):
    with pytest.raises(CompositionError) as e:
        compose.compose(seeded, "a blue apple")
    assert "not in blue" in e.value.missing[0] and "red" in e.value.missing[0]


def test_allow_partial_is_explicit_opt_in(seeded):
    r = compose.compose(seeded, "an apple and a banana", allow_partial=True)
    assert [o["label"] for o in r["image"]["objects"]] == ["apple"] and any("banana" in w for w in r["warnings"])


def test_relations_in_front_overlaps_and_orders_layers(seeded):
    r = compose.compose(seeded, "a red circle in front of a blue square")
    by = {p["label"]: p for p in r["placements"]}
    assert iou(by["circle"]["bbox"], by["square"]["bbox"]) > 0 and by["circle"]["z"] > by["square"]["z"]


def test_scale_and_position_follow_instruction(seeded):
    r = compose.compose(seeded, "a small apple to the left of a large orange")
    by = {p["label"]: p for p in r["placements"]}
    assert by["apple"]["bbox"][0] < by["orange"]["bbox"][0]
    assert by["orange"]["bbox"][3] > 1.5 * by["apple"]["bbox"][3]
    r = compose.compose(seeded, "a boy on the left, a girl on the right")
    by = {p["label"]: p for p in r["placements"]}
    assert by["boy"]["bbox"][0] < 300 < by["girl"]["bbox"][0]


def test_composites_are_not_reused_as_assets(seeded):
    compose.compose(seeded, "one apple and one orange")
    p = compose.plan(seeded, "an apple")
    assert all(a["image_name"] in ("apple", "apple_orange") for r in p["resolved"] for a in r["assets"])


def test_group_does_not_duplicate_a_shape_kind_after_manual_addition(seeded):
    lc = seeded.add_image(img_bytes("low_contrast"), "lc")
    seeded.add_object(lc["id"], "light gray circle", [100, 100, 200, 200])
    p = compose.plan(seeded, "colored shapes")
    labels = [a["label"] for a in p["resolved"][0]["assets"]]
    assert sorted(labels) == ["circle", "rectangle", "square", "triangle"]


def test_manual_correction_feeds_the_composer(seeded):
    """Detection missed it -> user added it -> it is now a usable asset. A removed object is not."""
    lc = seeded.add_image(img_bytes("low_contrast"), "lc")
    with pytest.raises(CompositionError):
        compose.compose(seeded, "a pebble")
    seeded.add_object(lc["id"], "pebble", [100, 100, 200, 200])
    r = compose.compose(seeded, "a pebble and an apple")
    assert sorted(o["label"] for o in r["image"]["objects"]) == ["apple", "pebble"]
    pid = next(o["id"] for o in seeded.get_image(lc["id"])["objects"] if o["label"] == "pebble")
    seeded.delete_object(pid)
    with pytest.raises(CompositionError):
        compose.compose(seeded, "a pebble")


def test_transparent_background_has_alpha(seeded):
    r = compose.compose(seeded, "an apple on a transparent background")
    im = cv2.imdecode(np.frombuffer(seeded.image_path(r["image"]["id"]).read_bytes(), np.uint8), cv2.IMREAD_UNCHANGED)
    assert im.shape[2] == 4 and im[0, 0, 3] == 0 and im[..., 3].max() == 255


def test_two_of_the_same_asset_are_placed_apart(seeded):
    r = compose.compose(seeded, "two oranges")
    a, b = [p["bbox"] for p in r["placements"]]
    assert iou(a, b) == 0 and a[2] == b[2]


def test_llm_planner_cannot_invent_assets(seeded):
    fake = lambda prompt: '{"canvas":{"background":"white"},"items":[{"text":"a dragon","label":"dragon","count":1,"color":null,"size":1,"anchor":null,"group":false,"relation_to_previous":"adjacent"}]}'
    parsed = compose.llm_plan("make a dragon", seeded, transport=fake)
    assert parsed and parsed["_llm"]
    resolved, missing, _ = compose.resolve(parsed["items"], seeded.live_objects())
    assert missing and not resolved                                                 # validated by the same resolver


def test_llm_planner_falls_back_on_garbage(seeded):
    assert compose.llm_plan("x", seeded, transport=lambda p: "I cannot do that") is None
