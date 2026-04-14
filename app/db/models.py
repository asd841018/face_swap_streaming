from datetime import datetime
from typing import Optional

from sqlalchemy import DateTime, Enum as SAEnum, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column
from sqlalchemy.sql import func
from app.schemas.video import VideoJobStatus
from sqlalchemy.orm import DeclarativeBase


class Base(DeclarativeBase):
    pass


class VideoJob(Base):
    __tablename__ = "video_jobs"

    job_id: Mapped[str] = mapped_column(String(32), primary_key=True)
    owner_key: Mapped[str] = mapped_column(String(128), nullable=False, index=True)
    status: Mapped[VideoJobStatus] = mapped_column(
        SAEnum(VideoJobStatus, name="video_job_status", native_enum=True, values_callable=lambda e: [m.value for m in e]),
        nullable=False,
        default=VideoJobStatus.QUEUED,
        index=True,
    )
    image_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    video_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    output_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    callback_url: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    error: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    total_frames: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        onupdate=func.now(),
        nullable=False,
    )
