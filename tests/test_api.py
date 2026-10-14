import io
import pytest
from fastapi.testclient import TestClient
from app.api import create_app
from tests.conftest import img_bytes


@pytest.fixture()
def client(tmp_path):
    client = TestClient(create_app(tmp_path / "store"))
    client.post("/api/auth/register", json={"username": "tester", "display_name": "Test User", "password": "testpass123"})
    client.headers["Authorization"] = "Bearer " + client.post("/api/auth/login", json={"username": "tester", "password": "testpass123"}).json()["token"]
    return client


def up(client, name):
    return client.post("/api/images", files={"file": (f"{name}.png", img_bytes(name), "image/png")})


def test_upload_detect_and_metadata_shape(client):
    r = up(client, "shapes"); assert r.status_code == 201
    d = r.json()
    assert d["auto"]["object_count"] == 4 and set(d) >= {"auto", "user", "objects", "summary"}
    assert up(client, "shapes").status_code == 409
    assert client.get(f"/api/images/{d['id']}/metadata").json()["id"] == d["id"]
    assert client.get(f"/api/images/{d['id']}/file").status_code == 200 and client.get(f"/api/images/{d['id']}/thumb").status_code == 200


def test_manual_correction_roundtrip(client):
    d = up(client, "low_contrast").json(); iid = d["id"]
    o = client.post(f"/api/images/{iid}/objects", json={"label": "circle", "bbox": [100, 100, 200, 200], "confidence": 0.8}).json()
    p = client.patch(f"/api/objects/{o['id']}", json={"label": "disc", "color": "light gray"}).json()
    assert p["label"] == "disc" and p["field_source"]["label"] == "user"
    again = client.post(f"/api/images/{iid}/detect", json={"mode": "preserve_manual"}).json()
    assert [x["label"] for x in again["objects"]] == ["disc"]
    wiped = client.post(f"/api/images/{iid}/detect", json={"mode": "replace_all"}).json()
    assert wiped["objects"] == []


def test_detect_requires_explicit_mode(client):
    iid = up(client, "apple").json()["id"]
    assert client.post(f"/api/images/{iid}/detect", json={}).status_code == 422
    assert client.post(f"/api/images/{iid}/detect", json={"mode": "auto"}).status_code == 422


def test_error_mapping(client):
    assert client.get("/api/images/nope").status_code == 404
    assert client.patch("/api/objects/nope", json={"label": "x"}).status_code == 404
    iid = up(client, "apple").json()["id"]
    oid = client.get(f"/api/images/{iid}").json()["objects"][0]["id"]
    assert client.patch(f"/api/objects/{oid}", json={"confidence": 3}).status_code == 422
    assert client.post("/api/images", files={"file": ("x.png", b"junk", "image/png")}).status_code == 400


def test_compose_endpoints(client):
    assert client.post("/api/seed").json()["added"]
    plan = client.post("/api/compose/plan", json={"instruction": "one red apple and one orange"}).json()
    assert not plan["missing"] and len(plan["resolved"]) == 2
    r = client.post("/api/compose", json={"instruction": "one red apple and one orange", "width": 800, "height": 600})
    assert r.status_code == 201 and r.json()["image"]["auto"]["width"] == 800
    bad = client.post("/api/compose", json={"instruction": "a unicorn"})
    assert bad.status_code == 422 and bad.json()["missing"]


def test_health_and_index(client):
    assert client.get("/api/health").json()["ok"] and "Image catalog" in client.get("/").text
