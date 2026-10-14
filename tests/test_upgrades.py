"""Automated test suite covering the upgraded features:
1. Three image-generation modes (assets-only, hybrid, ai-generated)
2. Intelligent prompt understanding (Gemini Flash & heuristic planner)
3. YOLO-World detector & benchmark comparison endpoint
4. Background editing tools (blur, replace, rembg cutout)
5. User-adjustable confidence threshold on redetection
"""
import cv2
import numpy as np
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
    return client.post("/api/images", files={"file": (f"{name}.png", img_bytes(name), "image/png")}).json()


# -----------------------------------------------------------------------------
# 1. Benchmark Endpoint
# -----------------------------------------------------------------------------
def test_benchmark_endpoint(client):
    r = client.get("/api/benchmark")
    assert r.status_code == 200
    data = r.json()
    assert "comparison" in data
    assert "hybrid_default" in data["comparison"]
    assert "yolo_world" in data["comparison"]
    assert "classical_only" in data["comparison"]
    assert "recommendation" in data
    h = data["comparison"]["hybrid_default"]
    assert h["precision"] >= 0.9 and h["recall"] >= 0.9


# -----------------------------------------------------------------------------
# 2. Planner & Intent Interpretation
# -----------------------------------------------------------------------------
def test_planner_mode_determination(client):
    # Upload catalog assets first
    up(client, "apple")
    up(client, "orange")

    # Mode 1: Assets only
    r1 = client.post("/api/planner/interpret", json={"instruction": "Create an image containing one red apple and one orange"}).json()
    assert r1["plan"]["generation_mode"] == "assets_only"

    # Mode 2: Hybrid (Asset + AI background)
    r2 = client.post("/api/planner/interpret", json={"instruction": "Place an apple on a futuristic cyberpunk background"}).json()
    assert r2["plan"]["generation_mode"] == "hybrid"
    assert "cyberpunk" in r2["plan"]["ai_generation"]["prompt"].lower() or "futuristic" in r2["plan"]["ai_generation"]["prompt"].lower()

    # Mode 3: Pure AI synthesis (concepts not in catalog)
    r3 = client.post("/api/planner/interpret", json={"instruction": "Generate a girl wearing a red hat, standing in a garden"}).json()
    assert r3["plan"]["generation_mode"] == "ai_generated"


def test_planner_clarification(client):
    r = client.post("/api/planner/interpret", json={"instruction": "Place it on the left"}).json()
    plan = r["plan"]
    # Vague prompt triggers clarification or asks which asset
    assert plan["clarification_needed"] is True
    assert len(plan["clarification_options"]) > 0


# -----------------------------------------------------------------------------
# 3. Three Generation Modes in /api/compose
# -----------------------------------------------------------------------------
def test_mode_1_assets_only(client):
    up(client, "apple")
    up(client, "orange")
    r = client.post("/api/compose", json={
        "instruction": "one red apple and one orange",
        "mode": "assets_only"
    })
    assert r.status_code == 201
    data = r.json()
    assert data["mode"] == "assets_only"
    assert len(data["placements"]) == 2


def test_mode_2_hybrid(client):
    up(client, "apple")
    r = client.post("/api/compose", json={
        "instruction": "place an apple on a futuristic dark background",
        "mode": "hybrid",
        "ai_style": "cartoon"
    })
    assert r.status_code == 201
    data = r.json()
    assert data["mode"] == "hybrid"
    assert len(data["placements"]) >= 1


def test_mode_3_ai_generated(client):
    r = client.post("/api/compose", json={
        "instruction": "Generate a girl wearing a red hat, standing in a garden",
        "mode": "ai_generated",
        "ai_style": "illustration"
    })
    assert r.status_code == 201
    data = r.json()
    assert data["mode"] == "ai_generated"
    assert data["image"]["user"]["display_name"]
    # Check that image was stored in catalog
    assert len(client.get("/api/images").json()) == 1


# -----------------------------------------------------------------------------
# 4. Background Editing (Blur, Replace, Rembg Cutout)
# -----------------------------------------------------------------------------
def test_background_editing_endpoints(client):
    img = up(client, "apple")
    img_id = img["id"]

    # Blur background
    r_blur = client.post(f"/api/images/{img_id}/edit", json={
        "op": "blur_background",
        "params": {"radius": 15},
        "mode": "copy"
    })
    assert r_blur.status_code == 201
    assert "background blurred" in r_blur.json()["image"]["user"]["display_name"]

    # Replace background with solid color
    r_rep = client.post(f"/api/images/{img_id}/edit", json={
        "op": "replace_background",
        "params": {"mode": "solid", "color": "#123456"},
        "mode": "copy"
    })
    assert r_rep.status_code == 201
    assert "background replaced" in r_rep.json()["image"]["user"]["display_name"]

    # Rembg background removal
    r_rembg = client.post(f"/api/images/{img_id}/edit", json={
        "op": "rembg_remove_bg",
        "params": {},
        "mode": "copy"
    })
    assert r_rembg.status_code == 201
    assert "AI background removed" in r_rembg.json()["image"]["user"]["display_name"]


# -----------------------------------------------------------------------------
# 5. Adjustable Confidence Threshold
# -----------------------------------------------------------------------------
def test_redetect_confidence_threshold(client):
    img = up(client, "shapes")
    img_id = img["id"]

    # High threshold
    r_high = client.post(f"/api/images/{img_id}/detect", json={
        "mode": "replace_all",
        "conf_threshold": 0.99
    })
    assert r_high.status_code == 200

    # Low threshold
    r_low = client.post(f"/api/images/{img_id}/detect", json={
        "mode": "replace_all",
        "conf_threshold": 0.10
    })
    assert r_low.status_code == 200
    assert len(r_low.json()["objects"]) >= len(r_high.json()["objects"])


# -----------------------------------------------------------------------------
# 6. Mode 2 Hybrid with Missing Catalog Assets (User's Exact Scenario)
# -----------------------------------------------------------------------------
def test_mode_2_hybrid_with_missing_catalog_assets(client):
    """When the user asks for 'two apples and one banana' in Mode 2, and the catalog
    only contains apple (not banana):
    - Must NOT fail with 422!
    - Must extract the 2 apples from catalog with locked layer fidelity.
    - Must synthesize the banana as an AI-generated element.
    """
    up(client, "apple")  # Catalog has apple, NO banana!

    r = client.post("/api/compose", json={
        "instruction": "two apples and one banana",
        "mode": "hybrid",
        "ai_style": "realistic"
    })
    assert r.status_code == 201
    data = r.json()
    assert data["mode"] == "hybrid"
    assert len(data["placements"]) == 2
    assert all(p["label"] == "apple" for p in data["placements"])
    assert any("banana" in w.lower() for w in data.get("warnings", []))


# -----------------------------------------------------------------------------
# 7. User Authentication & Profile API
# -----------------------------------------------------------------------------
def test_auth_login_and_user_stats(client):
    assert client.post("/api/auth/login", json={"username": "admin", "password": "admin123"}).status_code == 401
    assert client.post("/api/auth/login", json={"username": "tester", "password": "short"}).status_code == 401
    r_log = client.post("/api/auth/login", json={"username": "tester", "password": "testpass123"})
    assert r_log.status_code == 200
    token = r_log.json()["token"]
    r_me = client.get("/api/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert r_me.status_code == 200
    assert r_me.json()["user"]["username"] == "tester"
    assert "stats" in r_me.json()
    r_reg = client.post("/api/auth/register", json={
        "username": "ivan",
        "password": "password123",
        "display_name": "Ivan Ajay David",
    })
    assert r_reg.status_code == 201
    ivan_token = r_reg.json()["token"]
    assert ivan_token
    assert r_reg.json()["user"]["display_name"] == "Ivan Ajay David"
    r_up = client.post(
        "/api/images",
        files={"file": ("ivan_apple.png", img_bytes("apple"), "image/png")},
        headers={"Authorization": f"Bearer {ivan_token}"}
    )
    assert r_up.status_code == 201
    assert r_up.json()["user_id"] == r_reg.json()["user"]["id"]
    assert "auth_token" in r_reg.cookies and "httponly" in r_reg.headers.get("set-cookie", "").lower()
    assert client.get("/api/images").json() == []
    assert client.get("/api/images", headers={"Authorization": f"Bearer {ivan_token}"}).json()[0]["id"] == r_up.json()["id"]
    assert client.get(f"/api/images/{r_up.json()['id']}").status_code == 404
    assert client.get(f"/api/images/{r_up.json()['id']}", headers={"Authorization": f"Bearer {ivan_token}"}).status_code == 200
    media_client = TestClient(client.app)
    assert media_client.get("/api/auth/me", headers={"Authorization": f"Bearer {ivan_token}"}).status_code == 200
    assert media_client.cookies.get("auth_token") == ivan_token
    assert media_client.get(f"/api/images/{r_up.json()['id']}/thumb?size=120").status_code == 200
    assert media_client.get(f"/api/images/{r_up.json()['id']}/file").status_code == 200


def test_register_accepts_name_field_and_requires_a_name(client):
    r = client.post("/api/auth/register", json={"username": "nameduser", "name": "Named User", "password": "longpass123"})
    assert r.status_code == 201 and r.json()["user"]["display_name"] == "Named User"
    missing = client.post("/api/auth/register", json={"username": "noname", "password": "longpass123"})
    assert missing.status_code == 400 and "Name is required" in missing.json()["detail"]


def test_catalog_api_requires_auth_and_expired_tokens_fail(client):
    from datetime import datetime, timedelta, timezone
    token = client.headers["Authorization"].split(" ", 1)[1]
    client.headers.pop("Authorization")
    client.cookies.clear()
    assert client.get("/api/images").status_code == 401
    client.headers["Authorization"] = f"Bearer {token}"
    with client.app.state.auth._conn() as conn:
        expired = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat(timespec="seconds")
        conn.execute("UPDATE sessions SET expires_at=? WHERE token=?", (expired, token))
    assert client.get("/api/images").status_code == 401


# -----------------------------------------------------------------------------
# 8. Dedicated Image Generator Output Validation
# -----------------------------------------------------------------------------
def test_generator_local_scene_richness():
    from app.generator import IMAGE_GENERATOR

    # Test banana and apple semantic generation
    png, meta = IMAGE_GENERATOR._generate_local_scene("two apples and one banana", "realistic", 512, 384)
    assert len(png) > 1000
    assert meta["width"] == 512 and meta["height"] == 384

    # Test cartoon style
    png_c, meta_c = IMAGE_GENERATOR._generate_local_scene("a girl wearing a red hat in a garden", "cartoon", 512, 384)
    assert len(png_c) > 1000
    assert meta_c["style"] == "cartoon"
