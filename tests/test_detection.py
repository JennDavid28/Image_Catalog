"""Detection quality against exact ground truth (synthetic test catalog) and two real photographs."""
import cv2
import pytest

from app.detector import detect, iou
from app.evaluate import match, run
from tests.conftest import TESTSET, ROOT

CASES = ["apple", "orange", "boy", "girl", "shapes", "colors", "apple_orange", "overlapping_shapes"]


def _detect(name, **kw):
    return detect(cv2.imread(str(TESTSET / f"{name}.png")), **kw)


@pytest.mark.parametrize("name", CASES)
def test_ground_truth(name, testset):
    """Successful detection: right count, label, colour and tight boxes (IoU >= 0.9) with no extra objects."""
    gts = testset[name]
    res = _detect(name)
    pairs, missed, extra = match(res.objects, gts, thr=0.9)
    assert not missed, f"missed {[gts[j]['label'] for j in missed]}"
    assert not extra, f"spurious {[res.objects[i].label for i in extra]}"
    for i, j, _ in pairs:
        d, g = res.objects[i], gts[j]
        assert d.label == g["label"], (d.label, g["label"])
        assert d.color == g["color"], (d.color, g["color"])
        assert 0 <= d.confidence <= 1


def test_overlapping_boxes_overlap_and_are_full_extent(testset):
    """The circle is half hidden by the square, yet its box must cover the whole circle (amodal completion)."""
    res = _detect("overlapping_shapes")
    boxes = {d.label: d.bbox for d in res.objects}
    assert iou(boxes["circle"], boxes["square"]) > 0.0, "boxes should overlap"
    g = {o["label"]: o["bbox"] for o in testset["overlapping_shapes"]}
    assert iou(boxes["circle"], g["circle"]) > 0.95
    assert iou(boxes["triangle"], g["triangle"]) > 0.9       # apex hidden behind the square, restored


def test_five_colour_regions_have_distinct_names():
    res = _detect("colors")
    assert sorted(d.color for d in res.objects) == ["blue", "green", "purple", "red", "yellow"]
    assert all(d.label == "color swatch" for d in res.objects)


def test_cartoon_clothing_colours_reported():
    boy = _detect("boy").objects[0]
    girl = _detect("girl").objects[0]
    assert boy.attrs["clothing"]["top"] == "green" and boy.attrs["clothing"]["bottom"] == "blue"
    assert girl.attrs["clothing"]["top"] == "purple" and girl.attrs["clothing"]["bottom"] == "pink"


def test_uncertain_guesses_are_flagged_for_review():
    """boy/girl is inferred from the silhouette; the system must admit low certainty instead of faking it."""
    for n in ("boy", "girl"):
        r = _detect(n)
        assert r.objects[0].confidence < 0.6 and r.status == "low_confidence"
        assert any("low certainty" in e for e in r.objects[0].evidence)


def test_failed_detection_blank_image():
    r = _detect("blank")
    assert r.objects == [] and r.status == "no_objects" and r.message


def test_failed_detection_low_contrast_is_reported_not_hidden():
    """A genuinely invisible object is a documented miss: nothing returned, status says so (user adds it by hand)."""
    r = _detect("low_contrast")
    assert r.objects == [] and r.status == "no_objects"


def test_every_box_is_inside_the_image():
    for n in CASES:
        im = cv2.imread(str(TESTSET / f"{n}.png")); H, W = im.shape[:2]
        for d in detect(im).objects:
            x, y, w, h = d.bbox
            assert x >= 0 and y >= 0 and x + w <= W and y + h <= H and w > 0 and h > 0


def test_every_detection_explains_itself():
    for n in CASES:
        for d in _detect(n).objects:
            assert d.evidence, f"{n}/{d.label} has no evidence"


def test_aggregate_metrics():
    out = run()["summary"]
    assert out["precision"] == 1.0
    assert out["recall"] >= 0.94                 # 18/19: the single miss is the deliberately invisible circle
    assert out["label_accuracy"] == 1.0 and out["color_accuracy"] == 1.0
    assert out["mean_iou"] >= 0.97


neural = pytest.mark.skipif(not (ROOT / "models" / "yolox_nano.onnx").exists(), reason="neural model not downloaded")


@neural
@pytest.mark.parametrize("file,label,color", [("apple.jpg", "apple", "red"), ("orange.jpg", "orange", "orange")])
def test_real_photographs(file, label, color):
    """Photographic route: YOLOX-nano proposal + GrabCut mask + colour sanity check (it relabels apple/orange mixups)."""
    r = detect(cv2.imread(str(ROOT / "tests" / "real" / file)))
    top = max(r.objects, key=lambda d: d.confidence)
    assert (top.label, top.color) == (label, color)
    assert r.route == "complex-background"


@neural
def test_neural_backend_can_be_switched_off():
    r = detect(cv2.imread(str(ROOT / "tests" / "real" / "apple.jpg")), use_neural=False)
    assert r.objects == [] and r.status == "failed" and "neural backend is unavailable" in r.message
