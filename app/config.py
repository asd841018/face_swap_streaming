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
    SWAPPING_MODEL_PATH: Optional[str] = Field(
        default="/home/asd841018/face_swap_streaming/.assets/models/inswapper_128_fp16.onnx", 
        description="Path to the face swapping model"
    )
    
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

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
