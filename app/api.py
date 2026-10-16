"""FastAPI layer. Thin: validation + HTTP mapping only; all rules live in catalog.py / compose.py."""
from __future__ import annotations
import io
from pathlib import Path
from typing import Literal, Optional

import cv2
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel, Field

from . import compose, imageops
from .catalog import Catalog, CatalogError, Duplicate, NotFound

ROOT = Path(__file__).resolve().parent.parent
STATIC = ROOT / "static"
TESTSET = ROOT / "data" / "testset"
SEED_ORDER = ["apple", "orange", "boy", "girl", "shapes", "colors", "apple_orange", "overlapping_shapes"]
MAX_UPLOAD = 15 * 1024 * 1024


class ImageMeta(BaseModel):
    display_name: Optional[str] = Field(None, max_length=120)
    description: Optional[str] = Field(None, max_length=2000)
    tags: Optional[list[str]] = None


class NewObject(BaseModel):
    label: str = Field(..., min_length=1, max_length=80)
    bbox: list[float] = Field(..., min_length=4, max_length=4)
    color: Optional[str] = Field(None, max_length=60)
    confidence: Optional[float] = Field(None, ge=0, le=1)


class ObjectPatch(BaseModel):
    label: Optional[str] = Field(None, max_length=80)
    color: Optional[str] = Field(None, max_length=60)
    confidence: Optional[float] = Field(None, ge=0, le=1)
    bbox: Optional[list[float]] = Field(None, min_length=4, max_length=4)


class DetectReq(BaseModel):
    # explicit by design: the client must say what happens to manual work
    mode: Literal["replace_all", "preserve_manual"]
    extra_labels: Optional[list[str]] = Field(None, max_length=40)     # "also look for ..." (open-vocabulary backend only)
    conf_threshold: Optional[float] = Field(None, ge=0.01, le=1.0)


class EditReq(BaseModel):
    op: str = Field(..., max_length=40)
    params: dict = Field(default_factory=dict)
    mode: Literal["copy", "replace"] = "copy"
    name: Optional[str] = Field(None, max_length=120)


class ClassPatch(BaseModel):
    color: str = Field(..., max_length=9)


class ComposeReq(BaseModel):
    instruction: str = Field(..., min_length=3, max_length=1000)
    width: int = Field(1024, ge=128, le=2400)
    height: int = Field(768, ge=128, le=2400)
    background: Optional[str] = None
    allow_partial: bool = False
    name: Optional[str] = Field(None, max_length=120)
    llm: bool = False
    mode: Optional[Literal["assets_only", "hybrid", "ai_generated"]] = None
    ai_style: Optional[str] = Field("realistic", max_length=40)


class InterpretReq(BaseModel):
    instruction: str = Field(..., min_length=2, max_length=1000)
    model_name: Optional[str] = None


class GenerateReq(BaseModel):
    prompt: str = Field(..., min_length=3, max_length=1000)
    style: Optional[str] = "realistic"
    width: int = Field(1024, ge=128, le=2400)
    height: int = Field(768, ge=128, le=2400)
    name: Optional[str] = Field(None, max_length=120)


class LoginReq(BaseModel):
    # Credential mistakes should be reported as an authentication failure (401),
    # rather than rejected by request validation (422).
    username: str = Field(..., min_length=1, max_length=50)
    password: str = Field(..., max_length=100)


class RegisterReq(BaseModel):
    username: str = Field(..., min_length=1, max_length=50)
    display_name: Optional[str] = Field(None, max_length=80)
    name: Optional[str] = Field(None, max_length=80)
    password: str = Field(..., max_length=100)


def create_app(data_dir: str | Path | None = None) -> FastAPI:
    app = FastAPI(title="Image Catalog", version="1.0")
    cat = Catalog(data_dir)
    app.state.catalog = cat
    from .auth import AuthService
    auth_srv = AuthService(cat.db_path)
    app.state.auth = auth_srv

    @app.middleware("http")
    async def authenticated_catalog(request: Request, call_next):
        path = request.url.path
        if path.startswith("/api/") and path not in {
            "/api/auth/login", "/api/auth/register", "/api/health"
        }:
            header = request.headers.get("authorization", "")
            token = header[7:].strip() if header.lower().startswith("bearer ") else ""
            token = token or request.headers.get("x-auth-token", "") or request.cookies.get("auth_token", "")
            user = auth_srv.authenticate_token(token) if token else None
            if not user:
                return JSONResponse({"error": "Not authenticated"}, status_code=401)
            request.state.user = user

            parts = [part for part in path.split("/") if part]
            owner = None
            with cat._conn() as c:
                if len(parts) >= 3 and parts[1] == "images":
                    row = c.execute("SELECT user_id FROM images WHERE id=?", (parts[2],)).fetchone()
                    owner = row["user_id"] if row else None
                elif len(parts) >= 3 and parts[1] == "trash":
                    row = c.execute("SELECT user_id FROM images WHERE id=?", (parts[2],)).fetchone()
                    owner = row["user_id"] if row else None
                elif len(parts) >= 3 and parts[1] == "objects":
                    row = c.execute("SELECT i.user_id FROM objects o JOIN images i ON i.id=o.image_id WHERE o.id=?", (parts[2],)).fetchone()
                    owner = row["user_id"] if row else None
            if len(parts) >= 3 and parts[1] in {"images", "objects", "trash"} and owner != user["id"]:
                return JSONResponse({"error": "Not found"}, status_code=404)
        return await call_next(request)

    @app.exception_handler(NotFound)
    async def _nf(_: Request, e: NotFound):
        return JSONResponse({"error": str(e)}, 404)

    @app.exception_handler(Duplicate)
    async def _dup(_: Request, e: Duplicate):
        return JSONResponse({"error": str(e), "existing_id": e.existing_id}, 409)

    @app.exception_handler(CatalogError)
    async def _ce(_: Request, e: CatalogError):
        return JSONResponse({"error": str(e)}, 400)

    @app.exception_handler(compose.CompositionError)
    async def _comp(_: Request, e: compose.CompositionError):
        return JSONResponse({"error": str(e), "missing": e.missing, "suggestions": e.suggestions, "plan": e.plan}, 422)

    @app.post("/api/auth/register", status_code=201)
    def auth_register(body: RegisterReq, response: Response, req: Request):
        try:
            token, user = auth_srv.register(body.username, body.password, body.display_name or body.name)
            response.set_cookie("auth_token", token, max_age=30 * 24 * 60 * 60,
                                httponly=True, secure=req.url.scheme == "https",
                                samesite="lax", path="/api")
            return {"token": token, "user": user, "stats": auth_srv.get_user_stats(user["id"])}
        except ValueError as e:
            raise HTTPException(400, str(e))

    @app.post("/api/auth/login")
    def auth_login(body: LoginReq, response: Response, req: Request):
        try:
            token, user = auth_srv.login(body.username, body.password)
            response.set_cookie("auth_token", token, max_age=30 * 24 * 60 * 60,
                                httponly=True, secure=req.url.scheme == "https",
                                samesite="lax", path="/api")
            return {"token": token, "user": user, "stats": auth_srv.get_user_stats(user["id"])}
        except ValueError as e:
            raise HTTPException(401, str(e))

    @app.get("/api/auth/me")
    def auth_me(req: Request, response: Response):
        header = req.headers.get("authorization", "")
        token = (header[7:].strip() if header.lower().startswith("bearer ") else req.headers.get("x-auth-token", "")) or req.cookies.get("auth_token", "")
        user = getattr(req.state, "user", None)
        if not user:
            user = auth_srv.authenticate_token(token) if token else None
        if not user:
            raise HTTPException(401, "Not authenticated")
        # Upgrade sessions created before image media switched to cookie auth.
        response.set_cookie("auth_token", token, max_age=30 * 24 * 60 * 60,
                            httponly=True, secure=req.url.scheme == "https",
                            samesite="lax", path="/api")
        return {"user": user, "stats": auth_srv.get_user_stats(user["id"])}

    @app.post("/api/auth/logout")
    def auth_logout(req: Request, response: Response):
        header = req.headers.get("authorization", "")
        token = (header[7:].strip() if header.lower().startswith("bearer ") else req.headers.get("x-auth-token", "")) or req.cookies.get("auth_token", "")
        auth_srv.logout(token)
        response.delete_cookie("auth_token", path="/api", httponly=True, samesite="lax")
        return {"ok": True}

    # ------------------------------------------------------------------ images
    @app.get("/api/images")
    def list_images(req: Request):
        return cat.list_images(user_id=req.state.user["id"])

    @app.post("/api/images", status_code=201)
    async def upload(req: Request, file: UploadFile = File(...), display_name: Optional[str] = Form(None), detect: bool = Form(True)):
        data = await file.read()
        if not data:
            raise HTTPException(400, "empty file")
        if len(data) > MAX_UPLOAD:
            raise HTTPException(413, "file too large (15 MB max)")
        return cat.add_image(data, display_name, file.filename, detect=detect, user_id=req.state.user["id"])

    @app.get("/api/images/{image_id}")
    def get_image(image_id: str):
        return cat.get_image(image_id)

    @app.patch("/api/images/{image_id}")
    def patch_image(image_id: str, body: ImageMeta):
        return cat.update_image(image_id, body.display_name, body.description, body.tags)

    @app.delete("/api/images/{image_id}")
    def delete_image(image_id: str):
        """Moves the image to the trash folder (restorable). Use DELETE /api/trash/{id} to remove it for good."""
        return cat.trash_image(image_id)

    @app.post("/api/images/{image_id}/restore")
    def restore_image(image_id: str):
        return cat.restore_image(image_id)

    @app.get("/api/trash")
    def trash(req: Request):
        return cat.list_trash(user_id=req.state.user["id"])

    @app.delete("/api/trash/{image_id}")
    def purge(image_id: str):
        cat.purge_image(image_id)
        return {"purged": image_id}

    @app.delete("/api/trash")
    def empty_trash(req: Request):
        return {"purged": cat.empty_trash(user_id=req.state.user["id"])}

    @app.post("/api/images/{image_id}/edit", status_code=201)
    def edit_image(image_id: str, body: EditReq):
        return cat.edit_image(image_id, body.op, body.params, body.mode, body.name)

    @app.get("/api/images/{image_id}/polygons")
    def polygons(image_id: str):
        return cat.polygons(image_id)

    @app.get("/api/classes")
    def classes(req: Request):
        return cat.classes(user_id=req.state.user["id"])

    @app.patch("/api/classes/{label:path}")
    def class_color(label: str, body: ClassPatch, req: Request):
        try:
            return cat.set_class_color(label, body.color, user_id=req.state.user["id"])
        except imageops.EditError as e:
            raise HTTPException(422, str(e))

    @app.get("/api/capabilities")
    def capabilities():
        from .detector import DETECTOR_VERSION, _YOLOX
        from .openvocab import OPENVOCAB
        from .yoloworld import YOLOWORLD
        ov = OPENVOCAB.status()
        yw = YOLOWORLD.status()
        open_vocab_on = ov["available"] or yw["available"]
        return {"detector": DETECTOR_VERSION, "coco_model": _YOLOX.available(),
                "open_vocabulary": open_vocab_on,
                "yolo_world": yw["available"],
                "open_vocabulary_note": f"YOLO-World: {yw['reason']}; OWLv2: {ov['reason']}",
                "known_classes": "any word (open-vocabulary)" if open_vocab_on else "80 COCO classes plus cartoon shapes/fruit/people rules"}

    @app.get("/api/images/{image_id}/file")
    def image_file(image_id: str, download: bool = False):
        h = {"Cache-Control": "no-cache"}
        if download:
            p = cat.image_path(image_id)
            h["Content-Disposition"] = f'attachment; filename="{cat.get_image(image_id)["user"]["display_name"]}{p.suffix}"'.encode("ascii", "ignore").decode()
        return FileResponse(cat.image_path(image_id), headers=h)

    @app.get("/api/images/{image_id}/thumb")
    def thumb(image_id: str, size: int = 240):
        im = cat.read_bgr(image_id)
        s = min(1.0, max(32, min(size, 800)) / max(im.shape[:2]))
        im = cv2.resize(im, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)
        return Response(cv2.imencode(".jpg", im, [cv2.IMWRITE_JPEG_QUALITY, 85])[1].tobytes(), media_type="image/jpeg",
                        headers={"Cache-Control": "max-age=300"})

    @app.get("/api/images/{image_id}/events")
    def events(image_id: str):
        return cat.events(image_id)

    @app.get("/api/images/{image_id}/metadata")
    def metadata(image_id: str):
        """Full JSON export (auto vs user fields, per-object provenance)."""
        img = cat.get_image(image_id)
        return JSONResponse(img, headers={"Content-Disposition": f'attachment; filename="{image_id}.json"'})

    @app.post("/api/images/{image_id}/detect")
    def redetect(image_id: str, body: DetectReq):
        return cat.detect(image_id, body.mode, body.extra_labels, conf_threshold=body.conf_threshold)

    # ------------------------------------------------------------------ objects
    @app.post("/api/images/{image_id}/objects", status_code=201)
    def add_object(image_id: str, body: NewObject):
        return cat.add_object(image_id, body.label, body.bbox, body.color, body.confidence)

    @app.patch("/api/objects/{obj_id}")
    def patch_object(obj_id: str, body: ObjectPatch):
        return cat.update_object(obj_id, body.model_dump(exclude_unset=True))

    @app.delete("/api/objects/{obj_id}")
    def delete_object(obj_id: str):
        return cat.delete_object(obj_id)

    @app.post("/api/objects/{obj_id}/restore")
    def restore(obj_id: str):
        return cat.restore_object(obj_id)

    @app.post("/api/objects/{obj_id}/revert")
    def revert(obj_id: str):
        return cat.revert_object(obj_id)

    @app.post("/api/objects/{obj_id}/confirm")
    def confirm(obj_id: str, reviewed: bool = True):
        return cat.confirm_object(obj_id, reviewed)

    @app.get("/api/objects/{obj_id}/mask")
    def mask_png(obj_id: str):
        return Response(cv2.imencode(".png", cat.object_mask(obj_id))[1].tobytes(), media_type="image/png")

    @app.get("/api/objects/{obj_id}/asset")
    def asset(obj_id: str):
        a = cat.object_asset(obj_id)
        return Response(cv2.imencode(".png", a)[1].tobytes(), media_type="image/png")

    # ------------------------------------------------------------------ intelligent planner & generator
    @app.post("/api/planner/interpret")
    def planner_interpret(body: InterpretReq, req: Request):
        from .planner import plan_prompt
        # Let plan_prompt resolve GEMINI_MODEL from process settings or .env
        # unless this request explicitly asks for a particular model.
        p = plan_prompt(body.instruction, cat, model_name=body.model_name, user_id=req.state.user["id"])
        return {"plan": p, **p}

    @app.post("/api/generator/generate", status_code=201)
    def generate_image_api(body: GenerateReq, req: Request):
        return compose.compose(cat, body.prompt, width=body.width, height=body.height,
                               name=body.name, mode="ai_generated", ai_style=body.style or "realistic",
                               user_id=req.state.user["id"])

    @app.get("/api/benchmark")
    def benchmark_api():
        from .evaluate import benchmark_models
        return benchmark_models()

    # ------------------------------------------------------------------ composition
    @app.post("/api/compose/plan")
    def compose_plan(body: ComposeReq, req: Request):
        p = compose.plan(cat, body.instruction, llm=body.llm, mode=body.mode, user_id=req.state.user["id"])
        return {k: v for k, v in p.items() if not k.startswith("_")}

    @app.post("/api/compose", status_code=201)
    def compose_run(body: ComposeReq, req: Request):
        return compose.compose(cat, body.instruction, body.width, body.height, body.background,
                               body.allow_partial, body.name, body.llm, mode=body.mode,
                               ai_style=body.ai_style or "realistic", user_id=req.state.user["id"])

    # ------------------------------------------------------------------ demo helpers
    @app.post("/api/seed")
    def seed(req: Request):
        """Load the assignment's test catalog (skips ones already present)."""
        added, skipped = [], []
        for name in SEED_ORDER + ["blank", "low_contrast"]:
            p = TESTSET / f"{name}.png"
            if not p.exists():
                continue
            try:
                cat.add_image(p.read_bytes(), name, p.name, user_id=req.state.user["id"])
                added.append(name)
            except Duplicate:
                skipped.append(name)
        return {"added": added, "skipped": skipped}

    @app.get("/api/health")
    def health():
        from .detector import DETECTOR_VERSION, _YOLOX
        return {"ok": True, "detector": DETECTOR_VERSION, "neural_model": _YOLOX.available()}

    @app.get("/")
    def index():
        return FileResponse(STATIC / "index.html", headers={"Cache-Control": "no-cache"})

    return app


app = create_app()
