import json
import os
from datetime import datetime
from typing import Dict, Optional, Any
from app.schemas.session import SessionConfig, StreamStatus
from app.core import logger

SESSION_FILE = "user_sessions.json"

class Session:
    """代表一個用戶的推流會話"""
    def __init__(
        self,
        session_id: str,
        api_key: str,
        api_secret: str,
        config: SessionConfig,
        status: StreamStatus = StreamStatus.PENDING
    ):
        self.session_id = session_id
        self.api_key = api_key
        self.api_secret = api_secret
        self.config = config
        self.status = status
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.stats = {
            "uptime_seconds": 0,
            "frames_processed": 0,
            "current_fps": 0
        }

    def to_dict(self) -> Dict:
        return {
            "session_id": self.session_id,
            "api_key": self.api_key,
            "api_secret": self.api_secret,
            "config": self.config.model_dump(),
            "status": self.status.value,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "stats": self.stats
        }

    @classmethod
    def from_dict(cls, data: Dict) -> "Session":
        session = cls(
            session_id=data["session_id"],
            api_key=data["api_key"],
            api_secret=data["api_secret"],
            config=SessionConfig(**data["config"]),
            status=StreamStatus(data["status"])
        )
        session.created_at = datetime.fromisoformat(data["created_at"])
        session.updated_at = datetime.fromisoformat(data["updated_at"])
        session.stats = data.get("stats", {})
        return session

    def update_config(self, **kwargs):
        """更新配置"""
        config_dict = self.config.model_dump()
        for key, value in kwargs.items():
            if value is not None and key in config_dict:
                config_dict[key] = value
        self.config = SessionConfig(**config_dict)
        self.updated_at = datetime.now()


class SessionManager:
    """管理所有用戶會話"""
    _instance = None

    def __init__(self):
        self.sessions: Dict[str, Session] = {}  # session_id -> Session
        self.path_to_session: Dict[str, str] = {}  # path -> session_id
        self._load_sessions()

    @classmethod
    def get_instance(cls) -> "SessionManager":
        if cls._instance is None:
            cls._instance = SessionManager()
        return cls._instance

    def _load_sessions(self):
        """從文件加載會話"""
        if os.path.exists(SESSION_FILE):
            try:
                with open(SESSION_FILE, 'r') as f:
                    data = json.load(f)
                    for session_data in data.get("sessions", []):
                        session = Session.from_dict(session_data)
                        self.sessions[session.session_id] = session
                        # 重建 path 映射
                        path = f"{session.api_key}/{session.api_secret}"
                        self.path_to_session[path] = session.session_id
                logger.info(f"[SessionManager] Loaded {len(self.sessions)} sessions")
            except Exception as e:
                logger.error(f"[SessionManager] Failed to load sessions: {e}")

    def _save_sessions(self):
        """保存會話到文件"""
        try:
            data = {
                "sessions": [s.to_dict() for s in self.sessions.values()],
                "updated_at": datetime.now().isoformat()
            }
            with open(SESSION_FILE, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"[SessionManager] Failed to save sessions: {e}")

    def create_session(
        self,
        api_key: str,
        api_secret: str,
        config: SessionConfig
    ) -> Session:
        """創建新會話"""
        session_id = f"{api_key}_{api_secret}"
        path = f"{api_key}/{api_secret}"
        
        # 如果已存在，更新配置
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session.config = config
            session.updated_at = datetime.now()
            logger.info(f"[SessionManager] Updated existing session: {session_id}")
        else:
            session = Session(
                session_id=session_id,
                api_key=api_key,
                api_secret=api_secret,
                config=config
            )
            self.sessions[session_id] = session
            self.path_to_session[path] = session_id
            logger.info(f"[SessionManager] Created new session: {session_id}")
        
        self._save_sessions()
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """獲取會話"""
        return self.sessions.get(session_id)

    def get_session_by_path(self, path: str) -> Optional[Session]:
        """通過 RTMP path 獲取會話"""
        session_id = self.path_to_session.get(path)
        if session_id:
            return self.sessions.get(session_id)
        return None

    def update_session(self, session_id: str, **kwargs) -> Optional[Session]:
        """更新會話配置"""
        session = self.sessions.get(session_id)
        if session:
            session.update_config(**kwargs)
            self._save_sessions()
            return session
        return None

    def update_session_status(self, session_id: str, status: StreamStatus) -> Optional[Session]:
        """更新會話狀態"""
        session = self.sessions.get(session_id)
        if session:
            session.status = status
            session.updated_at = datetime.now()
            self._save_sessions()
            return session
        return None

    def delete_session(self, session_id: str) -> bool:
        """刪除會話"""
        if session_id in self.sessions:
            session = self.sessions[session_id]
            path = f"{session.api_key}/{session.api_secret}"
            del self.sessions[session_id]
            if path in self.path_to_session:
                del self.path_to_session[path]
            self._save_sessions()
            logger.info(f"[SessionManager] Deleted session: {session_id}")
            return True
        return False

    def list_sessions(self, api_key: Optional[str] = None) -> list[Session]:
        """列出會話"""
        if api_key:
            return [s for s in self.sessions.values() if s.api_key == api_key]
        return list(self.sessions.values())

    def get_session_for_stream(self, path: str) -> Optional[Session]:
        """
        當 MediaMTX 收到推流時，根據 path 找到對應的會話配置
        path 格式: api_key/api_secret
        """
        session = self.get_session_by_path(path)
        if session:
            return session
        
        # 嘗試解析 path
        parts = path.strip('/').split('/')
        if len(parts) >= 2:
            api_key = parts[0]
            api_secret = parts[1]
            session_id = f"{api_key}_{api_secret}"
            return self.sessions.get(session_id)
        
        return None


# 全局實例
session_manager = SessionManager.get_instance()
