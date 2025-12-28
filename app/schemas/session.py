from pydantic import BaseModel, Field, HttpUrl
from typing import Optional, Dict, Any
from datetime import datetime
from enum import Enum

class StreamStatus(str, Enum):
    PENDING = "pending"        # 用戶已配置，等待推流
    ACTIVE = "active"          # 正在處理中
    STOPPED = "stopped"        # 已停止
    ERROR = "error"            # 錯誤狀態

class SessionConfig(BaseModel):
    """用戶推流前配置的設定"""
    output_url: str = Field(..., description="目標推流地址，例如 Twitch/YouTube RTMP URL")
    source_face_url: Optional[str] = Field(None, description="換臉來源圖片 URL")
    swap_all: bool = Field(default=False, description="是否替換畫面中所有人臉")
    video_bitrate: Optional[str] = Field(default="3000k", description="輸出視頻比特率")
    video_resolution: Optional[str] = Field(default="1280x720", description="輸出解析度")
    frame_rate: Optional[int] = Field(default=25, description="輸出幀率")

class SessionCreate(BaseModel):
    """創建新會話的請求"""
    api_key: str = Field(..., description="用戶 API Key")
    api_secret: str = Field(..., description="用戶 API Secret")
    config: SessionConfig

class SessionUpdate(BaseModel):
    """更新會話配置的請求"""
    output_url: Optional[str] = None
    source_face_url: Optional[str] = None
    swap_all: Optional[bool] = None
    video_bitrate: Optional[str] = None
    video_resolution: Optional[str] = None
    frame_rate: Optional[int] = None

class UpdateSourceFace(BaseModel):
    """更新換臉來源的請求"""
    source_face_url: str = Field(..., description="新的換臉來源圖片 URL")

class SessionResponse(BaseModel):
    """會話資訊響應"""
    session_id: str
    api_key: str
    status: StreamStatus
    input_rtmp_url: str
    output_url: str
    source_face_url: Optional[str] = None
    created_at: datetime
    updated_at: datetime
    config: Dict[str, Any]

class SessionListResponse(BaseModel):
    """會話列表響應"""
    sessions: list[SessionResponse]
    total: int

class StreamStats(BaseModel):
    """推流統計信息"""
    session_id: str
    status: StreamStatus
    uptime_seconds: Optional[float] = None
    frames_processed: Optional[int] = None
    current_fps: Optional[float] = None
    input_rtmp_url: str
    output_url: str

class ApiResponse(BaseModel):
    """通用 API 響應"""
    success: bool
    message: str
    data: Optional[Dict[str, Any]] = None
