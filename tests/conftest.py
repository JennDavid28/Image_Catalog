import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
TESTSET = ROOT / "data" / "testset"
SEED = ["apple", "orange", "boy", "girl", "shapes", "colors", "apple_orange", "overlapping_shapes"]


@pytest.fixture(scope="session", autouse=True)
def testset():
    if not (TESTSET / "ground_truth.json").exists():
        from scripts.make_testdata import build
        build()
    return json.loads((TESTSET / "ground_truth.json").read_text())


def img_bytes(name):
    return (TESTSET / f"{name}.png").read_bytes()


@pytest.fixture()
def cat(tmp_path):
    from app.catalog import Catalog
    return Catalog(tmp_path / "store")


@pytest.fixture()
def seeded(cat):
    for n in SEED:
        cat.add_image(img_bytes(n), n, f"{n}.png")
    return cat


def by_name(cat, name):
    return next(i for i in cat.list_images() if i["user"]["display_name"] == name)
