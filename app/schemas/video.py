from typing import Optional
from pydantic import BaseModel, HttpUrl

class VideoSwapRequest(BaseModel):
    """Request model for video face swap"""
    owner_key: str
    image_url: str  # Source face image URL
    video_url: str  # Video URL to process


class VideoSwapResponse(BaseModel):
    """Response model for video face swap"""
    success: bool
    message: str
    output_url: Optional[str] = None
    owner_key: Optional[str] = None