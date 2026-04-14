from typing import Optional
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    # Server Configuration
    HOST: Optional[str] = Field(default="0.0.0.0")
    PORT: Optional[int] = Field(default=7980)
    
    # Face Swapping Configuration
    FACE_ANALYSIS_NAME: Optional[str] = Field(
        default="buffalo_l", 
        description="Name of the face analysis model"
    )
    SWAPPING_MODEL_PATH: str = Field(
        default="", 
        description="Path to the face swapping model"
    )

    # Runtime / Stream Defaults
    MEDIAMTX_API_URL: str = Field(
        default="http://localhost:9997/v3/paths/list",
        description="MediaMTX API endpoint for active paths",
    )
    FACESWAP_API_BASE_URL: str = Field(
        default="",
        description="External faceswap API base URL",
    )
    MONITOR_POLL_INTERVAL_SECONDS: float = Field(
        default=2.0,
        description="Polling interval for stream monitor",
    )
    FACESWAP_API_TIMEOUT_SECONDS: float = Field(
        default=8.0,
        description="Timeout for external faceswap API calls",
    )
    MAX_WORKERS: int = Field(
        default=2,
        description="Maximum number of concurrent worker processes",
    )
    MAX_CONCURRENT_VIDEO_SWAP_JOBS: int = Field(
        default=1,
        description="Maximum number of video swap jobs allowed to run the model at the same time",
    )
    DEFAULT_VIDEO_BITRATE: str = Field(default="3000k", description="Default output bitrate")
    DEFAULT_VIDEO_RESOLUTION: str = Field(default="1280x720", description="Default output resolution")
    DEFAULT_FRAME_RATE: int = Field(default=12, description="Default output frame rate")
    
    # Source Face Configuration
    SOURCE_FACE_DIR: Optional[str] = Field(
        default="/home/asd841018/face_swap_streaming/.assets/source", 
        description="Directory containing source face images"
    )
    SOURCE_FACE: Optional[str] = Field(
        default="rose.jpeg", 
        description="Filename of the source face image"
    )
    
    # S3 Configuration
    S3_UPLOAD_BUCKET: str = Field(
        default="operator-intelligence-upload",
        description="S3 bucket for uploading files",
    )
    AWS_REGION: str = Field(default="ap-southeast-1", description="AWS region")
    AWS_ACCESS_KEY_ID: str = Field(default="", description="AWS access key ID")
    AWS_SECRET_ACCESS_KEY: str = Field(default="", description="AWS secret access key")
    
    # Face Adjustment Parameters
    CHEEK_STRENGTH: float = Field(default=0.10, description="Strength of cheek adjustment")
    CHIN_STRENGTH: float = Field(default=0.50, description="Strength of chin adjustment")
    GRID_RESOLUTION: int = Field(default=50, description="Resolution of the adjustment grid")

    # Filter URL fallback mapping (for compatibility with old payloads)
    FILTER_REF_URL_COLD: str = Field(
        default="https://gsiai-dev-leo.s3.ap-southeast-1.amazonaws.com/public-example/f6ff09ff-9255-473d-a76e-8b9dc8a05a32.png",
        description="Legacy ref image URL mapped to cold beauty filter",
    )
    FILTER_REF_URL_WARM: str = Field(
        default="https://gsiai-dev-leo.s3.ap-southeast-1.amazonaws.com/public-example/6b0bc0da-0047-4066-8f15-504945bf656a.png",
        description="Legacy ref image URL mapped to warm beauty filter",
    )
    FILTER_REF_URL_VINTAGE: str = Field(
        default="https://gsiai-dev-leo.s3.ap-southeast-1.amazonaws.com/public-example/d5871d46-e53a-48b7-a843-de48f683696c.png",
        description="Legacy ref image URL mapped to vintage filter",
    )

    # Streaming Encoder Configuration (FFMPEG or GStreamer)
    STREAM_ENCODER: str = Field(
        default="ffmpeg",
        description="Streaming encoder to use (e.g., ffmpeg, gstreamer)"
    )
    
    # Output rtmp server configuration
    OUTPUT_SERVER_IP: str = Field(
        default="",
        description="IP address of the RTMP server to push the processed stream to"
    )

    # Webhook Configuration (outbound progress callbacks)
    WEBHOOK_TIMEOUT_SECONDS: float = Field(default=10.0)
    WEBHOOK_MAX_RETRIES: int = Field(default=3)
    WEBHOOK_SIGNING_SECRET: str = Field(
        default="",
        description="If set, each webhook body is HMAC-SHA256 signed and the hex digest sent in X-Webhook-Signature",
    )

    # Postgres Configuration
    POSTGRES_USER: str = Field(default="ai_livestream_admin")
    POSTGRES_PASSWORD: str = Field(default="1234")
    POSTGRES_HOST: str = Field(default="localhost")
    POSTGRES_PORT: int = Field(default=5432)
    POSTGRES_DB: str = Field(default="ai_livestream")

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+asyncpg://{self.POSTGRES_USER}:{self.POSTGRES_PASSWORD}"
            f"@{self.POSTGRES_HOST}:{self.POSTGRES_PORT}/{self.POSTGRES_DB}"
        )

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
