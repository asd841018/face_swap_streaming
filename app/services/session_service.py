import json
import os
import tempfile
import threading
from datetime import datetime
from typing import Dict, Optional
from app.schemas.session import SessionConfig, StreamStatus
from app.core import logger

SESSION_FILE = "user_sessions.json"


class Session:
    """A single user streaming session."""

    def __init__(self, session_id: str, api_key: str, api_secret: str,
                 config: SessionConfig, status: StreamStatus = StreamStatus.PENDING):
        self.session_id = session_id
        self.api_key = api_key
        self.api_secret = api_secret
        self.config = config
        self.status = status
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.stats: Dict = {"uptime_seconds": 0, "frames_processed": 0, "current_fps": 0}

    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "api_key": self.api_key,
            "api_secret": self.api_secret,
            "config": self.config.model_dump(),
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "stats": self.stats,
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Session":
        s = cls(
            session_id=data["session_id"],
            api_key=data["api_key"],
            api_secret=data["api_secret"],
            config=SessionConfig(**data["config"]),
            status=StreamStatus(data["status"]),
        )
        s.created_at = datetime.fromisoformat(data["created_at"])
        s.updated_at = datetime.fromisoformat(data["updated_at"])
        s.stats = data.get("stats", {})
        return s

    def update_config(self, **kwargs):
        d = self.config.model_dump()
        for k, v in kwargs.items():
            if v is not None and k in d:
                d[k] = v
        self.config = SessionConfig(**d)
        self.updated_at = datetime.now()


class SessionManager:
    _instance = None
    _init_done = False

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if SessionManager._init_done:
            return
        SessionManager._init_done = True
        self.sessions: Dict[str, Session] = {}
        self.path_to_session: Dict[str, str] = {}
        self._lock = threading.Lock()
        self._load()

    @classmethod
    def get_instance(cls) -> "SessionManager":
        return cls()

    # --- Persistence ---

    def _load(self):
        if not os.path.exists(SESSION_FILE):
            return
        try:
            with open(SESSION_FILE, "r") as f:
                for sd in json.load(f).get("sessions", []):
                    s = Session.from_dict(sd)
                    self.sessions[s.session_id] = s
                    self.path_to_session[f"{s.api_key}/{s.api_secret}"] = s.session_id
            logger.info(f"[SessionManager] Loaded {len(self.sessions)} sessions")
        except Exception as e:
            logger.error(f"[SessionManager] Load failed: {e}")

    def _save(self):
        try:
            payload = {
                "sessions": [s.to_dict() for s in self.sessions.values()],
                "updated_at": datetime.now().isoformat(),
            }
            with self._lock:
                with tempfile.NamedTemporaryFile("w", delete=False, dir=".", suffix=".tmp") as tmp:
                    json.dump(payload, tmp, indent=2)
                    tmp_path = tmp.name
                os.replace(tmp_path, SESSION_FILE)
        except Exception as e:
            logger.error(f"[SessionManager] Save failed: {e}")

    # --- CRUD ---

    def create_session(self, api_key: str, api_secret: str, config: SessionConfig) -> Session:
        sid = f"{api_key}_{api_secret}"
        path = f"{api_key}/{api_secret}"
        if sid in self.sessions:
            s = self.sessions[sid]
            s.config = config
            s.updated_at = datetime.now()
        else:
            s = Session(session_id=sid, api_key=api_key, api_secret=api_secret, config=config)
            self.sessions[sid] = s
            self.path_to_session[path] = sid
        self._save()
        return s

    def get_session(self, session_id: str) -> Optional[Session]:
        return self.sessions.get(session_id)

    def get_session_by_path(self, path: str) -> Optional[Session]:
        sid = self.path_to_session.get(path)
        return self.sessions.get(sid) if sid else None

    def update_session(self, session_id: str, **kwargs) -> Optional[Session]:
        s = self.sessions.get(session_id)
        if s:
            s.update_config(**kwargs)
            self._save()
        return s

    def update_session_status(self, session_id: str, status: StreamStatus) -> Optional[Session]:
        s = self.sessions.get(session_id)
        if s:
            s.status = status
            s.updated_at = datetime.now()
            self._save()
        return s

    def delete_session(self, session_id: str) -> bool:
        s = self.sessions.pop(session_id, None)
        if not s:
            return False
        self.path_to_session.pop(f"{s.api_key}/{s.api_secret}", None)
        self._save()
        return True

    def list_sessions(self, api_key: Optional[str] = None) -> list[Session]:
        if api_key:
            return [s for s in self.sessions.values() if s.api_key == api_key]
        return list(self.sessions.values())

    def get_session_for_stream(self, path: str) -> Optional[Session]:
        """Find session by RTMP path (api_key/api_secret or api_key/api_secret/sub_index)."""
        s = self.get_session_by_path(path)
        if s:
            return s
        parts = path.strip("/").split("/")
        # For paths with sub-index (e.g. api_key/api_secret/001),
        # look up by base path api_key/api_secret
        if len(parts) >= 3:
            base_path = f"{parts[0]}/{parts[1]}"
            s = self.get_session_by_path(base_path)
            if s:
                return s
        if len(parts) >= 2:
            return self.sessions.get(f"{parts[0]}_{parts[1]}")
        return None


session_manager = SessionManager.get_instance()
