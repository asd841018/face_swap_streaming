import threading
from dataclasses import dataclass, field
from datetime import datetime, UTC
from typing import Dict, Optional

from app.schemas.video import VideoJobStatus, VideoSwapStatusResponse


def _utcnow() -> datetime:
    return datetime.now(UTC)


@dataclass
class VideoJob:
    job_id: str
    owner_key: str
    output_url: Optional[str] = None
    status: VideoJobStatus = VideoJobStatus.QUEUED
    progress_percentage: int = 0
    current_step: str = "queued"
    message: str = "Job queued."
    processed_frames: int = 0
    total_frames: Optional[int] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=_utcnow)
    updated_at: datetime = field(default_factory=_utcnow)

    def to_response(self) -> VideoSwapStatusResponse:
        return VideoSwapStatusResponse(
            job_id=self.job_id,
            owner_key=self.owner_key,
            status=self.status,
            progress_percentage=self.progress_percentage,
            current_step=self.current_step,
            message=self.message,
            processed_frames=self.processed_frames,
            total_frames=self.total_frames,
            output_url=self.output_url,
            error=self.error,
            created_at=self.created_at,
            updated_at=self.updated_at,
        )


class VideoJobManager:
    def __init__(self):
        self.jobs: Dict[str, VideoJob] = {}
        self._lock = threading.Lock()

    def create_job(self, job_id: str, owner_key: str, output_url: Optional[str] = None) -> VideoJob:
        job = VideoJob(job_id=job_id, owner_key=owner_key, output_url=output_url)
        with self._lock:
            self.jobs[job_id] = job
        return job

    def get_job(self, job_id: str) -> Optional[VideoJob]:
        with self._lock:
            return self.jobs.get(job_id)

    def update_job(
        self,
        job_id: str,
        *,
        status: Optional[VideoJobStatus] = None,
        progress_percentage: Optional[int] = None,
        current_step: Optional[str] = None,
        message: Optional[str] = None,
        processed_frames: Optional[int] = None,
        total_frames: Optional[int] = None,
        output_url: Optional[str] = None,
        error: Optional[str] = None,
    ) -> Optional[VideoJob]:
        with self._lock:
            job = self.jobs.get(job_id)
            if job is None:
                return None
            if status is not None:
                job.status = status
            if progress_percentage is not None:
                job.progress_percentage = max(0, min(100, progress_percentage))
            if current_step is not None:
                job.current_step = current_step
            if message is not None:
                job.message = message
            if processed_frames is not None:
                job.processed_frames = processed_frames
            if total_frames is not None:
                job.total_frames = total_frames
            if output_url is not None:
                job.output_url = output_url
            if error is not None:
                job.error = error
            job.updated_at = _utcnow()
            return job

    def update_processing_progress(self, job_id: str, processed_frames: int, total_frames: int) -> Optional[VideoJob]:
        if total_frames > 0:
            progress = 40 + int((processed_frames / total_frames) * 50)
        else:
            progress = 40
        return self.update_job(
            job_id,
            status=VideoJobStatus.PROCESSING,
            progress_percentage=min(progress, 90),
            current_step="processing_video",
            message=f"Processing video frames: {processed_frames}/{total_frames or '?'}",
            processed_frames=processed_frames,
            total_frames=total_frames or None,
        )

    def mark_completed(self, job_id: str, output_url: Optional[str] = None) -> Optional[VideoJob]:
        return self.update_job(
            job_id,
            status=VideoJobStatus.COMPLETED,
            progress_percentage=100,
            current_step="completed",
            message="Video swap completed.",
            output_url=output_url,
            error=None,
        )

    def mark_failed(self, job_id: str, error: str) -> Optional[VideoJob]:
        return self.update_job(
            job_id,
            status=VideoJobStatus.FAILED,
            current_step="failed",
            message="Video swap failed.",
            error=error,
        )


video_job_manager = VideoJobManager()
