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
        default="/home/asd841018/detection/.assets/models/dynamic_batch_model.onnx", 
        description="Path to the face swapping model"
    )
    
    # Source Face Configuration
    SOURCE_FACE_DIR: Optional[str] = Field(
        default="/home/asd841018/detection/.assets/source", 
        description="Directory containing source face images"
    )
    SOURCE_FACE: Optional[str] = Field(
        default="rose.jpeg", 
        description="Filename of the source face image"
    )

    class Config:
        env_file = ".env"
        case_sensitive = True


settings = Settings()
