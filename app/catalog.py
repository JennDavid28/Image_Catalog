"""Catalog service: SQLite metadata + files on disk.

Source-of-truth rules (the heart of the assignment):
  * Every object field (label / color / confidence / bbox) carries a provenance: 'auto' or 'user'.
  * Original auto values are snapshotted (auto_json + <id>.auto.png) so a user edit can always be reverted.
  * Deleting an auto object leaves a tombstone, so a later re-detection does not resurrect it.
  * Re-detection only ever runs when explicitly requested, in one of two named modes:
        replace_all     - wipe everything (including manual work) and start from fresh detections
        preserve_manual - only refresh untouched auto objects; keep manual/edited/reviewed objects + tombstones
  * Every change is written to an audit log (events).
"""
from __future__ import annotations
import hashlib
import io
import json
import os
import shutil
import sqlite3
import threading
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from . import detector, imageops
from .detector import NEEDS_REVIEW, iou
from .segmentation import mask_bbox, segment_box
from .colors import region_color

ROOT = Path(__file__).resolve().parent.parent
FIELDS = ("label", "color", "confidence", "bbox")
SCHEMA = """
CREATE TABLE IF NOT EXISTS images(
  id TEXT PRIMARY KEY, kind TEXT NOT NULL DEFAULT 'source', filename TEXT, sha256 TEXT, width INT, height INT,
  format TEXT, bytes INT, created_at TEXT,
  display_name TEXT, description TEXT DEFAULT '', tags TEXT DEFAULT '[]',
  detection_status TEXT DEFAULT 'pending', detector_version TEXT, detected_at TEXT, route TEXT, detection_message TEXT,
  detection_ms REAL, dominant_colors TEXT DEFAULT '[]', provenance TEXT);
CREATE TABLE IF NOT EXISTS objects(
  id TEXT PRIMARY KEY, image_id TEXT NOT NULL REFERENCES images(id) ON DELETE CASCADE,
  label TEXT, color TEXT, confidence REAL, x INT, y INT, w INT, h INT,
  origin TEXT NOT NULL, field_src TEXT, auto_json TEXT, attrs TEXT DEFAULT '{}', evidence TEXT DEFAULT '[]',
  reviewed INT DEFAULT 0, deleted INT DEFAULT 0, mask_source TEXT, created_at TEXT, updated_at TEXT);
CREATE INDEX IF NOT EXISTS ix_obj_img ON objects(image_id);
CREATE TABLE IF NOT EXISTS events(
  id INTEGER PRIMARY KEY AUTOINCREMENT, image_id TEXT, object_id TEXT, ts TEXT, action TEXT, detail TEXT);
CREATE INDEX IF NOT EXISTS ix_evt_img ON events(image_id);
CREATE TABLE IF NOT EXISTS classes(label TEXT PRIMARY KEY, color TEXT NOT NULL);
"""
PALETTE = ["#6c63ff", "#22c55e", "#f97316", "#06b6d4", "#eab308", "#ef4444", "#a855f7", "#3b82f6", "#ec4899", "#14b8a6",
           "#84cc16", "#f43f5e", "#8b5cf6", "#0ea5e9", "#d946ef", "#10b981", "#f59e0b", "#64748b"]


class CatalogError(Exception):
    pass


class NotFound(CatalogError):
    pass


class Duplicate(CatalogError):
    def __init__(self, existing_id):
        super().__init__(f"identical image already in catalog ({existing_id})")
        self.existing_id = existing_id


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _uid() -> str:
    return uuid.uuid4().hex[:10]


class Catalog:
    def __init__(self, data_dir: str | Path | None = None):
        self.dir = Path(data_dir or os.environ.get("CATALOG_DATA") or ROOT / "data" / "store")
        (self.dir / "images").mkdir(parents=True, exist_ok=True)
        (self.dir / "masks").mkdir(parents=True, exist_ok=True)
        self.db_path = self.dir / "catalog.db"
        self._lock = threading.RLock()
        with self._conn() as c:
            c.executescript(SCHEMA)
            cols = {r["name"] for r in c.execute("PRAGMA table_info(images)")}
            if "trashed_at" not in cols:                        # migrate catalogs created before the trash folder existed
                c.execute("ALTER TABLE images ADD COLUMN trashed_at TEXT")
            if "user_id" not in cols:
                c.execute("ALTER TABLE images ADD COLUMN user_id TEXT DEFAULT 'user-admin'")

    # ------------------------------------------------------------------ plumbing
    @contextmanager
    def _conn(self):
        with self._lock:
            c = sqlite3.connect(self.db_path)
            c.row_factory = sqlite3.Row
            c.execute("PRAGMA foreign_keys=ON")
            try:
                yield c
                c.commit()
            finally:
                c.close()

    def _log(self, c, image_id, object_id, action, detail=None):
        c.execute("INSERT INTO events(image_id,object_id,ts,action,detail) VALUES(?,?,?,?,?)",
                  (image_id, object_id, _now(), action, json.dumps(detail or {})))

    def _img_row(self, c, image_id):
        r = c.execute("SELECT * FROM images WHERE id=?", (image_id,)).fetchone()
        if not r:
            raise NotFound(f"image {image_id} not found")
        return r

    def _obj_row(self, c, obj_id):
        r = c.execute("SELECT * FROM objects WHERE id=?", (obj_id,)).fetchone()
        if not r:
            raise NotFound(f"object {obj_id} not found")
        return r

    def image_path(self, image_id) -> Path:
        with self._conn() as c:
            r = self._img_row(c, image_id)
        return self.dir / "images" / f"{image_id}.{r['format']}"

    def mask_path(self, obj_id, auto=False) -> Path:
        return self.dir / "masks" / f"{obj_id}{'.auto' if auto else ''}.png"

    def read_bgr(self, image_id) -> np.ndarray:
        return _decode(self.image_path(image_id).read_bytes())

    # ------------------------------------------------------------------ serialisation
    def _obj_dict(self, r) -> dict:
        src = json.loads(r["field_src"] or "{}")
        auto = json.loads(r["auto_json"]) if r["auto_json"] else None
        conf = r["confidence"]
        needs = bool(r["origin"] == "auto" and not r["reviewed"] and conf is not None and conf < NEEDS_REVIEW
                     and not any(v == "user" for v in src.values()))
        return {"id": r["id"], "image_id": r["image_id"], "label": r["label"], "color": r["color"],
                "confidence": conf, "bbox": [r["x"], r["y"], r["w"], r["h"]], "origin": r["origin"],
                "field_source": src, "auto_values": auto, "reviewed": bool(r["reviewed"]), "needs_review": needs,
                "deleted": bool(r["deleted"]), "attrs": json.loads(r["attrs"] or "{}"),
                "evidence": json.loads(r["evidence"] or "[]"), "mask_source": r["mask_source"],
                "edited": any(v == "user" for v in src.values()), "created_at": r["created_at"], "updated_at": r["updated_at"]}

    def _img_dict(self, c, r, with_objects=True) -> dict:
        objs = [self._obj_dict(o) for o in c.execute(
            "SELECT * FROM objects WHERE image_id=? ORDER BY y, x", (r["id"],)).fetchall()]
        live = [o for o in objs if not o["deleted"]]
        d = {"id": r["id"], "kind": r["kind"],
             "user_id": r["user_id"] if "user_id" in r.keys() else "user-admin",
             "auto": {"width": r["width"], "height": r["height"], "format": r["format"], "bytes": r["bytes"],
                      "sha256": r["sha256"], "created_at": r["created_at"], "filename": r["filename"],
                      "detection_status": r["detection_status"], "detector_version": r["detector_version"],
                      "detected_at": r["detected_at"], "route": r["route"], "detection_message": r["detection_message"],
                      "detection_ms": r["detection_ms"], "trashed_at": r["trashed_at"], "dominant_colors": json.loads(r["dominant_colors"] or "[]"),
                      "object_count": len(live), "provenance": json.loads(r["provenance"]) if r["provenance"] else None},
             "user": {"display_name": r["display_name"], "description": r["description"], "tags": json.loads(r["tags"] or "[]")},
             "summary": {"needs_review": sum(o["needs_review"] for o in live),
                         "manually_corrected": any(o["origin"] == "manual" or o["edited"] for o in objs),
                         "deleted_objects": sum(o["deleted"] for o in objs)}}
        if with_objects:
            d["objects"] = objs
        return d

    # ------------------------------------------------------------------ images
    def add_image(self, data: bytes, display_name: str | None = None, filename: str | None = None,
                  detect: bool = True, kind: str = "source", provenance: dict | None = None,
                  user_id: str | None = None) -> dict:
        sha = hashlib.sha256(data).hexdigest()
        try:
            pil = Image.open(io.BytesIO(data))
            fmt = (pil.format or "PNG").lower().replace("jpeg", "jpg")
            pil.verify()
        except Exception as e:
            raise CatalogError(f"not a readable image: {e}")
        bgr = _decode(data)
        h, w = bgr.shape[:2]
        with self._conn() as c:
            if kind == "source":
                dup = c.execute("SELECT id FROM images WHERE sha256=? AND kind='source'", (sha,)).fetchone()
                if dup:
                    raise Duplicate(dup["id"])
            image_id = _uid()
            name = display_name or (Path(filename).stem if filename else f"image-{image_id[:4]}")
            c.execute("INSERT INTO images(id,kind,filename,sha256,width,height,format,bytes,created_at,display_name,provenance,user_id)"
                      " VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                      (image_id, kind, filename, sha, w, h, fmt, len(data), _now(), name,
                       json.dumps(provenance) if provenance else None, user_id or "user-admin"))
            (self.dir / "images" / f"{image_id}.{fmt}").write_bytes(data)
            self._log(c, image_id, None, "image_added", {"filename": filename, "kind": kind})
        if detect:
            self.detect(image_id, "initial")
        return self.get_image(image_id)

    def get_image(self, image_id) -> dict:
        with self._conn() as c:
            return self._img_dict(c, self._img_row(c, image_id))

    def list_images(self, user_id: str | None = None) -> list[dict]:
        with self._conn() as c:
            if user_id:
                rows = c.execute(
                    "SELECT * FROM images WHERE trashed_at IS NULL AND (user_id=? OR kind='source') ORDER BY created_at, rowid",
                    (user_id,)
                ).fetchall()
            else:
                rows = c.execute("SELECT * FROM images WHERE trashed_at IS NULL ORDER BY created_at, rowid").fetchall()
            return [self._img_dict(c, r, with_objects=False) for r in rows]

    def update_image(self, image_id, display_name=None, description=None, tags=None) -> dict:
        with self._conn() as c:
            self._img_row(c, image_id)
            before = {}
            for col, val in (("display_name", display_name), ("description", description),
                             ("tags", json.dumps(list(tags)) if tags is not None else None)):
                if val is not None:
                    c.execute(f"UPDATE images SET {col}=? WHERE id=?", (val, image_id))
                    before[col] = val
            self._log(c, image_id, None, "image_meta_edit", before)
        return self.get_image(image_id)

    def delete_image(self, image_id):
        with self._conn() as c:
            r = self._img_row(c, image_id)
            ids = [o["id"] for o in c.execute("SELECT id FROM objects WHERE image_id=?", (image_id,))]
            c.execute("DELETE FROM images WHERE id=?", (image_id,))
            c.execute("DELETE FROM events WHERE image_id=?", (image_id,))
        for oid in ids:
            for a in (False, True):
                self.mask_path(oid, a).unlink(missing_ok=True)
        (self.dir / "images" / f"{image_id}.{r['format']}").unlink(missing_ok=True)

    def events(self, image_id, limit=200) -> list[dict]:
        with self._conn() as c:
            self._img_row(c, image_id)
            rows = c.execute("SELECT * FROM events WHERE image_id=? ORDER BY id DESC LIMIT ?", (image_id, limit)).fetchall()
        return [{"id": r["id"], "object_id": r["object_id"], "ts": r["ts"], "action": r["action"],
                 "detail": json.loads(r["detail"] or "{}")} for r in rows]

    # ------------------------------------------------------------------ detection
    def detect(self, image_id, mode: str, extra_labels: list | None = None, conf_threshold: float | None = None) -> dict:
        """mode: 'initial' | 'replace_all' | 'preserve_manual'  (re-runs are always explicit)."""
        if mode not in ("initial", "replace_all", "preserve_manual"):
            raise CatalogError("mode must be 'replace_all' or 'preserve_manual'")
        bgr = self.read_bgr(image_id)
        res = detector.detect(bgr, extra_labels=extra_labels, conf_threshold=conf_threshold)
        with self._conn() as c:
            self._img_row(c, image_id)
            old = c.execute("SELECT * FROM objects WHERE image_id=?", (image_id,)).fetchall()
            keep, removed = [], 0
            for o in old:
                d = self._obj_dict(o)
                protected = mode == "preserve_manual" and (o["origin"] != "auto" or d["edited"] or d["reviewed"] or o["deleted"])
                if protected:
                    keep.append(d)
                else:
                    c.execute("DELETE FROM objects WHERE id=?", (o["id"],))
                    for a in (False, True):
                        self.mask_path(o["id"], a).unlink(missing_ok=True)
                    removed += 1
            added = suppressed = 0
            for det in res.objects:
                if any(iou(det.bbox, k["bbox"]) > 0.5 for k in keep):
                    suppressed += 1            # covered by a manual correction / reviewed object / tombstone
                    continue
                self._insert_auto(c, image_id, det)
                added += 1
            c.execute("UPDATE images SET detection_status=?, detector_version=?, detected_at=?, route=?, detection_message=?,"
                      " detection_ms=?, dominant_colors=? WHERE id=?",
                      (res.status, detector.DETECTOR_VERSION, _now(), res.route, res.message, round(res.ms, 1),
                       json.dumps(res.dominant_colors), image_id))
            self._log(c, image_id, None, "detect", {"mode": mode, "status": res.status, "route": res.route, "added": added,
                                                    "removed_auto": removed, "kept_manual": len(keep), "suppressed": suppressed})
        return self.get_image(image_id)

    def _insert_auto(self, c, image_id, det):
        oid = _uid()
        auto = {"label": det.label, "color": det.color, "confidence": det.confidence, "bbox": list(det.bbox)}
        cv2.imwrite(str(self.mask_path(oid, True)), det.mask.astype(np.uint8) * 255)
        shutil.copyfile(self.mask_path(oid, True), self.mask_path(oid))
        attrs = {k: v for k, v in det.attrs.items()}
        c.execute("INSERT INTO objects(id,image_id,label,color,confidence,x,y,w,h,origin,field_src,auto_json,attrs,evidence,"
                  "mask_source,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                  (oid, image_id, det.label, det.color, det.confidence, *det.bbox, "auto",
                   json.dumps({f: "auto" for f in FIELDS}), json.dumps(auto), json.dumps(attrs), json.dumps(det.evidence),
                   "detector", _now(), _now()))
        return oid

    # ------------------------------------------------------------------ manual correction
    def add_object(self, image_id, label, bbox, color=None, confidence=None, attrs=None) -> dict:
        label = (label or "").strip()
        if not label:
            raise CatalogError("label is required")
        bgr = self.read_bgr(image_id)
        x, y, w, h = _clamp_bbox(bbox, bgr.shape)
        mask, msrc = segment_box(bgr, (x, y, w, h))
        src = {"label": "user", "confidence": "user", "bbox": "user", "color": "user"}
        if not color:                                    # fill the colour from the pixels, but mark it as auto-derived
            color, rgbv = region_color(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB), mask, erode=7)
            src["color"] = "auto"
        oid = _uid()
        cv2.imwrite(str(self.mask_path(oid)), mask.astype(np.uint8) * 255)
        with self._conn() as c:
            self._img_row(c, image_id)
            c.execute("INSERT INTO objects(id,image_id,label,color,confidence,x,y,w,h,origin,field_src,auto_json,attrs,evidence,"
                      "reviewed,mask_source,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                      (oid, image_id, label, color, 1.0 if confidence is None else _conf(confidence), x, y, w, h, "manual",
                       json.dumps(src), None, json.dumps(attrs or {}), json.dumps(["added manually by user"]), 1, msrc, _now(), _now()))
            self._log(c, image_id, oid, "object_added", {"label": label, "bbox": [x, y, w, h]})
            return self._obj_dict(self._obj_row(c, oid))

    def update_object(self, obj_id, patch: dict) -> dict:
        with self._conn() as c:
            r = self._obj_row(c, obj_id)
            cur = self._obj_dict(r)
            src = dict(cur["field_source"])
            new = {}
            if "label" in patch and patch["label"] is not None:
                if not str(patch["label"]).strip():
                    raise CatalogError("label cannot be empty")
                new["label"] = str(patch["label"]).strip()
            if "color" in patch and patch["color"] is not None:
                new["color"] = str(patch["color"]).strip()
            if "confidence" in patch and patch["confidence"] is not None:
                new["confidence"] = _conf(patch["confidence"])
            bgr = None
            if "bbox" in patch and patch["bbox"] is not None:
                bgr = self.read_bgr(r["image_id"])
                new["bbox"] = list(_clamp_bbox(patch["bbox"], bgr.shape))
            changes = {k: {"from": cur[k], "to": v} for k, v in new.items() if v != cur[k]}
            for k in changes:
                src[k] = "user"
            if "attrs" in patch and patch["attrs"] is not None:
                c.execute("UPDATE objects SET attrs=? WHERE id=?", (json.dumps(patch["attrs"]), obj_id))
                changes["attrs"] = {"to": patch["attrs"]}
            if changes:
                row = {**cur, **new}
                x, y, w, h = row["bbox"]
                c.execute("UPDATE objects SET label=?,color=?,confidence=?,x=?,y=?,w=?,h=?,field_src=?,updated_at=? WHERE id=?",
                          (row["label"], row["color"], row["confidence"], x, y, w, h, json.dumps(src), _now(), obj_id))
                if "bbox" in changes:                    # geometry changed -> recompute the cut-out mask
                    mask, msrc = segment_box(bgr, (x, y, w, h))
                    cv2.imwrite(str(self.mask_path(obj_id)), mask.astype(np.uint8) * 255)
                    c.execute("UPDATE objects SET mask_source=? WHERE id=?", (msrc, obj_id))
                self._log(c, r["image_id"], obj_id, "object_edit", changes)
            return self._obj_dict(self._obj_row(c, obj_id))

    def delete_object(self, obj_id) -> dict:
        with self._conn() as c:
            r = self._obj_row(c, obj_id)
            if r["origin"] == "auto":                    # tombstone: re-detection must not bring it back
                c.execute("UPDATE objects SET deleted=1, updated_at=? WHERE id=?", (_now(), obj_id))
                kind = "tombstoned"
            else:
                c.execute("DELETE FROM objects WHERE id=?", (obj_id,))
                for a in (False, True):
                    self.mask_path(obj_id, a).unlink(missing_ok=True)
                kind = "removed"
            self._log(c, r["image_id"], obj_id, "object_deleted", {"label": r["label"], "how": kind})
        return {"id": obj_id, "result": kind}

    def restore_object(self, obj_id) -> dict:
        with self._conn() as c:
            r = self._obj_row(c, obj_id)
            c.execute("UPDATE objects SET deleted=0, updated_at=? WHERE id=?", (_now(), obj_id))
            self._log(c, r["image_id"], obj_id, "object_restored", {"label": r["label"]})
            return self._obj_dict(self._obj_row(c, obj_id))

    def revert_object(self, obj_id) -> dict:
        with self._conn() as c:
            r = self._obj_row(c, obj_id)
            if not r["auto_json"]:
                raise CatalogError("manual objects have no automatic values to revert to")
            a = json.loads(r["auto_json"])
            x, y, w, h = a["bbox"]
            c.execute("UPDATE objects SET label=?,color=?,confidence=?,x=?,y=?,w=?,h=?,field_src=?,reviewed=0,deleted=0,"
                      "mask_source='detector',updated_at=? WHERE id=?",
                      (a["label"], a["color"], a["confidence"], x, y, w, h, json.dumps({f: "auto" for f in FIELDS}), _now(), obj_id))
            if self.mask_path(obj_id, True).exists():
                shutil.copyfile(self.mask_path(obj_id, True), self.mask_path(obj_id))
            self._log(c, r["image_id"], obj_id, "object_reverted", a)
            return self._obj_dict(self._obj_row(c, obj_id))

    def confirm_object(self, obj_id, reviewed=True) -> dict:
        """Human verified an auto object as-is (clears the needs-review flag; protects it from preserve_manual re-runs)."""
        with self._conn() as c:
            r = self._obj_row(c, obj_id)
            c.execute("UPDATE objects SET reviewed=?, updated_at=? WHERE id=?", (1 if reviewed else 0, _now(), obj_id))
            self._log(c, r["image_id"], obj_id, "object_confirmed" if reviewed else "object_unconfirmed", {})
            return self._obj_dict(self._obj_row(c, obj_id))

    # ------------------------------------------------------------------ composites
    def add_composite(self, png: bytes, name: str, provenance: dict, placements: list[dict], user_id: str | None = None) -> dict:
        """Save a generated composite as its own catalog entry. Objects come straight from the placement plan
        (exact boxes, confidence 1.0, origin='composite') together with the asset each one was cut from."""
        img = self.add_image(png, name, filename=None, detect=False, kind="composite", provenance=provenance, user_id=user_id)
        W, H = img["auto"]["width"], img["auto"]["height"]
        with self._conn() as c:
            c.execute("UPDATE images SET detection_status='composed', detector_version='composition-plan', detected_at=?,"
                      " route='composition', detection_message=? WHERE id=?",
                      (_now(), "objects registered from the composition plan (not detected)", img["id"]))
            for pl in placements:
                x, y, w, h = _clamp_bbox(pl["bbox"], (H, W))
                c.execute("INSERT INTO objects(id,image_id,label,color,confidence,x,y,w,h,origin,field_src,auto_json,attrs,evidence,"
                          "mask_source,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                          (_uid(), img["id"], pl["label"], pl["color"], 1.0, x, y, w, h, "composite",
                           json.dumps({f: "auto" for f in FIELDS}), None,
                           json.dumps({"asset_image_id": pl["asset_image_id"], "asset_object_id": pl["asset_object_id"],
                                       "asset_image_name": pl["asset_image_name"], "z": pl["z"], "scale": pl["scale"]}),
                           json.dumps([f"pasted from '{pl['asset_image_name']}' (object {pl['asset_object_id']})"]),
                           "none", _now(), _now()))
            self._log(c, img["id"], None, "composite_created", {"instruction": provenance.get("instruction"),
                                                                 "assets": provenance.get("assets")})
        return self.get_image(img["id"])

    # ------------------------------------------------------------------ assets for composition
    def live_objects(self) -> list[dict]:
        """All non-deleted objects joined with their image display name/kind (used by the composer)."""
        with self._conn() as c:
            rows = c.execute("SELECT o.*, i.display_name AS img_name, i.kind AS img_kind, i.width AS img_w, i.height AS img_h "
                             "FROM objects o JOIN images i ON i.id=o.image_id WHERE o.deleted=0 AND i.trashed_at IS NULL").fetchall()
        out = []
        for r in rows:
            d = self._obj_dict(r)
            d.update(image_name=r["img_name"], image_kind=r["img_kind"], image_size=[r["img_w"], r["img_h"]])
            out.append(d)
        return out

    def _clipped_mask(self, r, bgr) -> np.ndarray:
        """uint8 0/255 mask for an object row, clipped to its (possibly user-edited) box."""
        mp = self.mask_path(r["id"])
        mask = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE) if mp.exists() else None
        x, y, w, h = r["x"], r["y"], r["w"], r["h"]
        if mask is None or mask.shape != bgr.shape[:2] or (mask[y:y + h, x:x + w] > 0).sum() < 20:
            mask = (segment_box(bgr, (x, y, w, h))[0]).astype(np.uint8) * 255
        clip = np.zeros_like(mask)
        clip[y:y + h, x:x + w] = mask[y:y + h, x:x + w]
        return clip

    def object_mask(self, obj_id) -> np.ndarray:
        with self._conn() as c:
            r = self._obj_row(c, obj_id)
        return self._clipped_mask(r, self.read_bgr(r["image_id"]))

    def object_asset(self, obj_id) -> np.ndarray:
        """BGRA cut-out of an object (cropped to its box). Edge pixels are eroded 1px before feathering so the old
        background colour does not leave a halo when the asset is pasted on a different backdrop."""
        with self._conn() as c:
            r = self._obj_row(c, obj_id)
        bgr = self.read_bgr(r["image_id"])
        x, y, w, h = r["x"], r["y"], r["w"], r["h"]
        clip = self._clipped_mask(r, bgr)
        a = cv2.erode(clip, np.ones((3, 3), np.uint8))
        a = cv2.GaussianBlur(a, (0, 0), 0.8)
        bb = mask_bbox(a > 8) or (x, y, w, h)
        bx, by, bw, bh = bb
        out = np.dstack([bgr[by:by + bh, bx:bx + bw], a[by:by + bh, bx:bx + bw]])
        return out


    # ------------------------------------------------------------------ trash folder
    def trash_image(self, image_id) -> dict:
        """Soft delete: the image leaves the catalog (and composer) but can be restored until the trash is emptied."""
        with self._conn() as c:
            self._img_row(c, image_id)
            c.execute("UPDATE images SET trashed_at=? WHERE id=?", (_now(), image_id))
            self._log(c, image_id, None, "image_trashed", {})
        return {"id": image_id, "trashed": True}

    def restore_image(self, image_id) -> dict:
        with self._conn() as c:
            self._img_row(c, image_id)
            c.execute("UPDATE images SET trashed_at=NULL WHERE id=?", (image_id,))
            self._log(c, image_id, None, "image_restored", {})
        return self.get_image(image_id)

    def list_trash(self, user_id: str | None = None) -> dict:
        with self._conn() as c:
            if user_id:
                imgs = [self._img_dict(c, r, with_objects=False) for r in
                        c.execute("SELECT * FROM images WHERE trashed_at IS NOT NULL AND (user_id=? OR user_id IS NULL) ORDER BY trashed_at DESC", (user_id,)).fetchall()]
                objs = []
                for r in c.execute("SELECT o.*, i.display_name AS img_name FROM objects o JOIN images i ON i.id=o.image_id "
                                   "WHERE o.deleted=1 AND i.trashed_at IS NULL AND (i.user_id=? OR i.user_id IS NULL) ORDER BY o.updated_at DESC", (user_id,)).fetchall():
                    d = self._obj_dict(r); d["image_name"] = r["img_name"]; objs.append(d)
            else:
                imgs = [self._img_dict(c, r, with_objects=False) for r in
                        c.execute("SELECT * FROM images WHERE trashed_at IS NOT NULL ORDER BY trashed_at DESC").fetchall()]
                objs = []
                for r in c.execute("SELECT o.*, i.display_name AS img_name FROM objects o JOIN images i ON i.id=o.image_id "
                                   "WHERE o.deleted=1 AND i.trashed_at IS NULL ORDER BY o.updated_at DESC").fetchall():
                    d = self._obj_dict(r); d["image_name"] = r["img_name"]; objs.append(d)
        return {"images": imgs, "objects": objs}

    def purge_image(self, image_id):
        """Permanent delete. Only allowed for images that are already in the trash."""
        with self._conn() as c:
            r = self._img_row(c, image_id)
            if not r["trashed_at"]:
                raise CatalogError("move the image to the trash first")
        self.delete_image(image_id)

    def empty_trash(self, user_id: str | None = None) -> int:
        ids = [i["id"] for i in self.list_trash(user_id=user_id)["images"]]
        for i in ids:
            self.delete_image(i)
        return len(ids)

    # ------------------------------------------------------------------ classes (label colours shown on the canvas)
    def _class_color(self, c, label: str) -> str:
        r = c.execute("SELECT color FROM classes WHERE label=?", (label,)).fetchone()
        if r:
            return r["color"]
        used = {x["color"] for x in c.execute("SELECT color FROM classes")}
        free = [p for p in PALETTE if p not in used]
        color = free[0] if free else PALETTE[int(hashlib.md5(label.encode()).hexdigest(), 16) % len(PALETTE)]
        c.execute("INSERT INTO classes(label,color) VALUES(?,?)", (label, color))
        return color

    def classes(self) -> list[dict]:
        """Every label in use (non-trashed images), with its overlay colour and how many objects carry it."""
        with self._conn() as c:
            rows = c.execute("SELECT o.label AS label, COUNT(*) AS n FROM objects o JOIN images i ON i.id=o.image_id "
                             "WHERE o.deleted=0 AND i.trashed_at IS NULL GROUP BY o.label ORDER BY n DESC, o.label").fetchall()
            return [{"label": r["label"], "count": r["n"], "color": self._class_color(c, r["label"])} for r in rows]

    def set_class_color(self, label: str, color: str) -> dict:
        r, g, b_ = imageops.hex_to_rgb(color)                       # validates 3- and 6-digit hex
        color = f"#{r:02x}{g:02x}{b_:02x}"
        with self._conn() as c:
            c.execute("INSERT INTO classes(label,color) VALUES(?,?) ON CONFLICT(label) DO UPDATE SET color=excluded.color",
                      (label, color))
        return {"label": label, "color": color}

    # ------------------------------------------------------------------ mask outlines for the translucent overlay
    def polygons(self, image_id) -> dict:
        """{object_id: {'color': '#hex', 'polys': [[[x,y],...], ...]}} for every non-deleted object."""
        bgr = self.read_bgr(image_id)
        out = {}
        with self._conn() as c:
            self._img_row(c, image_id)
            rows = c.execute("SELECT * FROM objects WHERE image_id=? AND deleted=0", (image_id,)).fetchall()
            for r in rows:
                color = self._class_color(c, r["label"])
                x, y, w, h = r["x"], r["y"], r["w"], r["h"]
                m = self._clipped_mask(r, bgr)
                s = min(1.0, 640 / max(m.shape))
                ms = cv2.resize(m, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else m
                cnts, _ = cv2.findContours((ms > 127).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:3]
                polys = []
                for cn in cnts:
                    if cv2.contourArea(cn) < 12 * s * s * 4:
                        continue
                    ap = cv2.approxPolyDP(cn, 0.004 * cv2.arcLength(cn, True), True).reshape(-1, 2)
                    if len(ap) >= 3:
                        polys.append([[round(float(px) / s, 1), round(float(py) / s, 1)] for px, py in ap])
                out[r["id"]] = {"color": color, "polys": polys or [[[x, y], [x + w, y], [x + w, y + h], [x, y + h]]]}
        return out

    IMAGE_OPS = ("crop", "resize", "rotate", "flip", "adjust", "recolor_object", "cut_object", "erase_object", "remove_background", "blur_background", "replace_background", "rembg_remove_bg")

    def edit_image(self, image_id, op: str, params: dict | None = None, mode: str = "copy", name: str | None = None) -> dict:
        """Apply an edit and save it. mode='copy' (default) leaves the original untouched and adds a new catalog entry;
        mode='replace' overwrites the original pixels (its objects are re-detected). Returns {'image': ..., 'op': ...}."""
        params = params or {}
        if op not in self.IMAGE_OPS:
            raise CatalogError(f"unknown edit '{op}'. Use one of: {', '.join(self.IMAGE_OPS)}")
        if mode not in ("copy", "replace"):
            raise CatalogError("mode must be 'copy' or 'replace'")
        with self._conn() as c:
            src = self._img_row(c, image_id)
        bgr = self.read_bgr(image_id)
        fmt = "png"
        try:
            if op == "crop":
                res = imageops.crop(bgr, params.get("x"), params.get("y"), params.get("width"), params.get("height"))
            elif op == "resize":
                res = imageops.resize(bgr, params.get("width"), params.get("height"), params.get("percent"))
            elif op == "rotate":
                res = imageops.rotate(bgr, params.get("degrees", 90))
            elif op == "flip":
                res = imageops.flip(bgr, params.get("axis", "horizontal"))
            elif op == "adjust":
                res = imageops.adjust(bgr, **{k: params[k] for k in ("hue", "saturation", "brightness", "contrast", "grayscale", "sepia", "invert") if k in params})
            elif op == "remove_background":
                ids = params.get("object_ids") or [o["id"] for o in self.get_image(image_id)["objects"] if not o["deleted"]]
                res = imageops.remove_background(bgr, [self.object_mask(i) for i in ids]); fmt = "png"
            elif op == "blur_background":
                ids = params.get("object_ids") or [o["id"] for o in self.get_image(image_id)["objects"] if not o["deleted"]]
                res = imageops.blur_background(bgr, [self.object_mask(i) for i in ids], ksize=params.get("ksize", params.get("radius", 25)))
            elif op == "replace_background":
                ids = params.get("object_ids") or [o["id"] for o in self.get_image(image_id)["objects"] if not o["deleted"]]
                bg_val = params.get("color") or params.get("background") or "#ffffff"
                res = imageops.replace_background(bgr, [self.object_mask(i) for i in ids], bg_val)
            elif op == "rembg_remove_bg":
                res = imageops.rembg_auto_remove_bg(bgr); fmt = "png"
            else:
                oid = params.get("object_id")
                if not oid:
                    raise CatalogError("object_id is required")
                with self._conn() as c:
                    orow = self._obj_row(c, oid)
                if orow["image_id"] != image_id:
                    raise CatalogError("that object belongs to a different image")
                m = self._clipped_mask(orow, bgr)
                if op == "recolor_object":
                    res = imageops.recolor_region(bgr, m, params.get("color", ""))
                elif op == "cut_object":
                    res = imageops.cutout(bgr, m, int(params.get("pad", 0)))
                else:
                    res = imageops.erase(bgr, m)
        except imageops.EditError as e:
            raise CatalogError(str(e))
        data = imageops.encode(res, fmt if op in ("cut_object", "remove_background", "rembg_remove_bg") else (src["format"] if src["format"] in ("png", "jpg", "webp") else "png"))
        label = {"crop": "cropped", "resize": "resized", "rotate": "rotated", "flip": "flipped", "adjust": "recoloured",
                 "recolor_object": "recoloured", "cut_object": "cut-out", "erase_object": "object erased",
                 "remove_background": "background removed", "blur_background": "background blurred",
                 "replace_background": "background replaced", "rembg_remove_bg": "AI background removed"}[op]
        if mode == "replace" and op not in ("cut_object",):
            return {"image": self._replace_pixels(image_id, data, op), "op": op, "mode": mode}
        new_name = name or f"{src['display_name']} ({label})"
        prov = {"edited_from": image_id, "edited_from_name": src["display_name"], "op": op,
                "params": {k: v for k, v in params.items() if k != "object_ids"}}
        try:
            img = self.add_image(data, new_name, filename=f"{image_id}-{op}.png", provenance=prov)
        except Duplicate as e:
            raise CatalogError(f"the result is identical to an image already in the catalog ({e.existing_id})")
        with self._conn() as c:
            self._log(c, image_id, None, "image_edited", {"op": op, "result": img["id"], "mode": "copy"})
        return {"image": img, "op": op, "mode": "copy"}

    def _replace_pixels(self, image_id, data: bytes, op: str) -> dict:
        pil = Image.open(io.BytesIO(data)); fmt = (pil.format or "PNG").lower().replace("jpeg", "jpg")
        bgr = _decode(data); h, w = bgr.shape[:2]
        with self._conn() as c:
            r = self._img_row(c, image_id)
            old = self.dir / "images" / f"{image_id}.{r['format']}"
            new = self.dir / "images" / f"{image_id}.{fmt}"
            new.write_bytes(data)
            if old != new:
                old.unlink(missing_ok=True)
            c.execute("UPDATE images SET sha256=?, width=?, height=?, format=?, bytes=? WHERE id=?",
                      (hashlib.sha256(data).hexdigest(), w, h, fmt, len(data), image_id))
            self._log(c, image_id, None, "image_edited", {"op": op, "mode": "replace"})
        return self.detect(image_id, "replace_all")


# ---------------------------------------------------------------------- helpers
def _decode(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    im = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)
    if im is None:
        raise CatalogError("could not decode image")
    if im.ndim == 2:
        return cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    if im.shape[2] == 4:                                  # flatten transparency over white
        a = im[..., 3:4].astype(np.float32) / 255
        return (im[..., :3] * a + 255 * (1 - a)).astype(np.uint8)
    return im


def _clamp_bbox(bbox, shape):
    try:
        x, y, w, h = [int(round(float(v))) for v in bbox]
    except Exception:
        raise CatalogError("bbox must be [x, y, width, height]")
    H, W = shape[:2]
    x, y = max(0, min(x, W - 2)), max(0, min(y, H - 2))
    w, h = max(2, min(w, W - x)), max(2, min(h, H - y))
    return x, y, w, h


def _conf(v) -> float:
    v = float(v)
    if not 0 <= v <= 1:
        raise CatalogError("confidence must be between 0 and 1")
    return round(v, 3)
