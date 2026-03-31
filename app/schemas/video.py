from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class VideoJobStatus(str, Enum):
    """Background video swap job status."""
    QUEUED = "queued"
    DOWNLOADING = "downloading"
    PROCESSING = "processing"
    UPLOADING = "uploading"
    COMPLETED = "completed"
    FAILED = "failed"

class VideoSwapRequest(BaseModel):
    """Request model for video face swap"""
    owner_key: str
    image_url: str  # Source face image URL
    video_url: str  # Video URL to process


class VideoSwapResponse(BaseModel):
    """Response model for video face swap"""
    success: bool
    message: str
    job_id: str
    output_url: Optional[str] = None
    status_url: str
    owner_key: Optional[str] = None


class VideoSwapStatusResponse(BaseModel):
    """Status model for polling background video swap progress."""
    job_id: str
    owner_key: str
    status: VideoJobStatus
    progress_percentage: int = Field(default=0, ge=0, le=100)
    current_step: str
    message: str
    processed_frames: int = 0
    total_frames: Optional[int] = None
    output_url: Optional[str] = None
    error: Optional[str] = None
    created_at: datetime
    updated_at: datetime
