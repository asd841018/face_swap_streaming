"""DB-backed CRUD for video swap jobs.

Persists only the skeleton fields (status, output_url, error, total_frames,
timestamps). High-frequency progress updates are NOT written here — they live
in ProgressBroadcaster and are pushed to WebSocket subscribers.
"""
from typing import Optional

from sqlalchemy import select

from app.db.models import VideoJob
from app.core.session import AsyncSessionLocal
from app.schemas.video import VideoJobStatus


class VideoJobManager:
    """Async Postgres-backed repository for video jobs."""

    async def create_job(
        self,
        job_id: str,
        owner_key: str,
        output_url: Optional[str] = None,
        callback_url: Optional[str] = None,
    ) -> VideoJob:
        async with AsyncSessionLocal() as session:
            job = VideoJob(
                job_id=job_id,
                owner_key=owner_key,
                output_url=output_url,
                callback_url=callback_url,
                status=VideoJobStatus.QUEUED,
            )
            session.add(job)
            await session.commit()
            await session.refresh(job)
            return job

    async def get_job(self, job_id: str) -> Optional[VideoJob]:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(VideoJob).where(VideoJob.job_id == job_id)
            )
            return result.scalar_one_or_none()

    async def update_job(
        self,
        job_id: str,
        *,
        status: Optional[VideoJobStatus] = None,
        output_url: Optional[str] = None,
        error: Optional[str] = None,
        total_frames: Optional[int] = None,
    ) -> Optional[VideoJob]:
        async with AsyncSessionLocal() as session:
            result = await session.execute(
                select(VideoJob).where(VideoJob.job_id == job_id)
            )
            job = result.scalar_one_or_none()
            if job is None:
                return None
            if status is not None:
                job.status = status
            if output_url is not None:
                job.output_url = output_url
            if error is not None:
                job.error = error
            if total_frames is not None:
                job.total_frames = total_frames
            await session.commit()
            await session.refresh(job)
            return job

    async def mark_completed(
        self, job_id: str, output_url: Optional[str] = None
    ) -> Optional[VideoJob]:
        return await self.update_job(
            job_id, status=VideoJobStatus.COMPLETED, output_url=output_url
        )

    async def mark_failed(self, job_id: str, error: str) -> Optional[VideoJob]:
        return await self.update_job(job_id, status=VideoJobStatus.FAILED, error=error)


video_job_manager = VideoJobManager()
