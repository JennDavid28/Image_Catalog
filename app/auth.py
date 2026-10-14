"""User Authentication & Profile Management Service.

Provides:
- User accounts with secure password hashing.
- Token-based session management (persistent in localStorage).
- Per-user data isolation and statistics.
"""
from __future__ import annotations
import hashlib
import secrets
import sqlite3
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

ROOT = Path(__file__).resolve().parent.parent

AUTH_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id TEXT PRIMARY KEY,
    username TEXT UNIQUE NOT NULL COLLATE NOCASE,
    password_hash TEXT NOT NULL,
    salt TEXT NOT NULL,
    display_name TEXT NOT NULL,
    email TEXT DEFAULT '',
    role TEXT DEFAULT 'Designer',
    avatar_color TEXT DEFAULT '#6c63ff',
    created_at TEXT NOT NULL,
    last_login TEXT
);

CREATE TABLE IF NOT EXISTS sessions (
    token TEXT PRIMARY KEY,
    user_id TEXT NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    created_at TEXT NOT NULL,
    expires_at TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS ix_sessions_user ON sessions(user_id);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _hash_password(password: str, salt: str) -> str:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt.encode("utf-8"), 100_000).hex()


AVATAR_COLORS = [
    "#6c63ff", "#3b82f6", "#10b981", "#f59e0b", "#ef4444",
    "#8b5cf6", "#ec4899", "#06b6d4", "#14b8a6", "#f97316"
]


class AuthService:
    def __init__(self, db_path: Path | None = None):
        self.db_path = db_path or (ROOT / "data" / "store" / "catalog.db")
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._init_db()

    def _conn(self):
        c = sqlite3.connect(self.db_path)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA foreign_keys=ON")
        return c

    def _init_db(self):
        with self._lock:
            with self._conn() as c:
                c.executescript(AUTH_SCHEMA)
                # Remove legacy, publicly documented demo accounts from older databases.
                c.execute("DELETE FROM users WHERE id IN ('user-admin', 'user-demo') OR username IN ('admin', 'demo')")
    def register(self, username: str, password: str, display_name: str | None = None,
                 email: str | None = None, role: str = "Designer") -> Tuple[str, Dict[str, Any]]:
        username = username.strip().lower()
        if len(username) < 3:
            raise ValueError("Username must be at least 3 characters")
        if len(password) < 8:
            raise ValueError("Password must be at least 8 characters")

        disp = (display_name or "").strip()
        if not disp:
            raise ValueError("Name is required")
        color = secrets.choice(AVATAR_COLORS)
        salt = secrets.token_hex(16)
        pw_hash = _hash_password(password, salt)
        user_id = f"user-{secrets.token_hex(6)}"

        with self._lock:
            with self._conn() as c:
                existing = c.execute("SELECT id FROM users WHERE username=?", (username,)).fetchone()
                if existing:
                    raise ValueError(f"Username '{username}' already exists")
                c.execute(
                    "INSERT INTO users (id, username, password_hash, salt, display_name, email, role, avatar_color, created_at, last_login) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (user_id, username, pw_hash, salt, disp, email or "", "Designer", color, _now(), _now())
                )
                token = self._create_session(c, user_id)
                user = self._user_dict(c.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone())
                return token, user

    def login(self, username: str, password: str) -> Tuple[str, Dict[str, Any]]:
        username = username.strip().lower()
        with self._lock:
            with self._conn() as c:
                row = c.execute("SELECT * FROM users WHERE username=?", (username,)).fetchone()
                if not row:
                    raise ValueError("Invalid username or password")
                expected = _hash_password(password, row["salt"])
                if expected != row["password_hash"]:
                    raise ValueError("Invalid username or password")

                c.execute("UPDATE users SET last_login=? WHERE id=?", (_now(), row["id"]))
                token = self._create_session(c, row["id"])
                user = self._user_dict(row)
                return token, user

    def _create_session(self, conn, user_id: str) -> str:
        token = secrets.token_hex(32)
        created = datetime.now(timezone.utc)
        expires = created + timedelta(days=30)
        conn.execute(
            "INSERT INTO sessions (token, user_id, created_at, expires_at) VALUES (?, ?, ?, ?)",
            (token, user_id, created.isoformat(timespec="seconds"), expires.isoformat(timespec="seconds"))
        )
        return token

    def authenticate_token(self, token: str) -> Optional[Dict[str, Any]]:
        if not token:
            return None
        with self._lock:
            with self._conn() as c:
                row = c.execute(
                    "SELECT u.*, s.expires_at FROM sessions s JOIN users u ON s.user_id = u.id WHERE s.token=?",
                    (token,)
                ).fetchone()
                if not row:
                    return None
                if datetime.fromisoformat(row["expires_at"]) <= datetime.now(timezone.utc):
                    c.execute("DELETE FROM sessions WHERE token=?", (token,))
                    return None
                return self._user_dict(row)

    def logout(self, token: str):
        if not token:
            return
        with self._lock:
            with self._conn() as c:
                c.execute("DELETE FROM sessions WHERE token=?", (token,))

    def get_user_stats(self, user_id: str) -> Dict[str, Any]:
        with self._lock:
            with self._conn() as c:
                # Images total
                images_count = c.execute(
                    "SELECT COUNT(*) as c FROM images WHERE trashed_at IS NULL AND user_id=?",
                    (user_id,)
                ).fetchone()["c"]
                # Composites created
                comp_count = c.execute(
                    "SELECT COUNT(*) as c FROM images WHERE kind='composite' AND trashed_at IS NULL AND user_id=?",
                    (user_id,)
                ).fetchone()["c"]
                # Objects count
                obj_count = c.execute(
                    "SELECT COUNT(*) as c FROM objects o JOIN images i ON o.image_id=i.id WHERE o.deleted=0 AND i.user_id=?",
                    (user_id,)
                ).fetchone()["c"]
                # Trashed count
                trash_count = c.execute(
                    "SELECT COUNT(*) as c FROM images WHERE trashed_at IS NOT NULL AND user_id=?",
                    (user_id,)
                ).fetchone()["c"]
                return {
                    "images_count": images_count,
                    "composites_count": comp_count,
                    "objects_count": obj_count,
                    "trash_count": trash_count
                }

    def _user_dict(self, row) -> Dict[str, Any]:
        if not row:
            return {}
        return {
            "id": row["id"],
            "username": row["username"],
            "display_name": row["display_name"],
            "email": row["email"],
            "role": row["role"],
            "avatar_color": row["avatar_color"],
            "created_at": row["created_at"],
            "last_login": row["last_login"]
        }
