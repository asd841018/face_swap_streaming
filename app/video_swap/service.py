"""Video face-swap service — orchestrates job queueing and background processing."""
import os
import uuid
import tempfile
import asyncio
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

from fastapi import BackgroundTasks

from app.core import logger
from app.config import settings
from app.schemas.video import (
    VideoJobStatus,
    VideoSwapRequest,
    VideoSwapResponse,
    VideoSwapStatusResponse,
)
from app.services.progress_broadcaster import progress_broadcaster
from app.services.video_job_service import video_job_manager

from app.video_swap.utils import (
    build_local_output_url,
    build_output_url,
    download_file,
    process_video_face_swap,
    upload_to_s3,
)
from app.video_swap.webhook import webhook_delivery


def _webhook_event(status: VideoJobStatus) -> str:
    if status == VideoJobStatus.COMPLETED:
        return "job.completed"
    if status == VideoJobStatus.FAILED:
        return "job.failed"
    return "job.updated"


class VideoSwapService:
    """Encapsulates queueing and background execution of video face-swap jobs."""

    def __init__(self) -> None:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        self.output_dir = os.path.join(current_dir, "swapped_videos")
        os.makedirs(self.output_dir, exist_ok=True)

        self._executor = ThreadPoolExecutor(max_workers=settings.MAX_WORKERS)
        self._processing_semaphore = asyncio.Semaphore(
            max(1, settings.MAX_CONCURRENT_VIDEO_SWAP_JOBS)
        )

    async def queue_swap_job(
        self,
        request: VideoSwapRequest,
        background_tasks: BackgroundTasks,
    ) -> VideoSwapResponse:
        job_id = str(uuid.uuid4())[:8]
        temp_dir = tempfile.mkdtemp(prefix=f"faceswap_{job_id}_")
        output_filename = f"{request.owner_key}_{job_id}_swapped.mp4"
        output_path = os.path.join(self.output_dir, output_filename)
        output_url = build_output_url(request.owner_key, output_filename)
        status_url = f"/api/video/swap/{job_id}/status"

        await video_job_manager.create_job(
            job_id=job_id,
            owner_key=request.owner_key,
            output_url=output_url,
            callback_url=request.callback_url,
        )

        await self._publish(
            job_id,
            request.callback_url,
            status=VideoJobStatus.QUEUED,
            progress_percentage=0,
            current_step="queued",
            message="Job queued.",
            owner_key=request.owner_key,
        )

        background_tasks.add_task(
            self._process_in_background,
            job_id,
            request.owner_key,
            request.image_url,
            request.video_url,
            output_path,
            temp_dir,
            request.callback_url,
        )

        logger.info(f"[VideoSwap:{job_id}] Job queued, returning immediately")

        return VideoSwapResponse(
            success=True,
            message=f"Video swap job {job_id} queued successfully. Processing in background.",
            job_id=job_id,
            output_url=output_url,
            status_url=status_url,
            owner_key=request.owner_key,
        )

    async def get_job_status(self, job_id: str) -> Optional[VideoSwapStatusResponse]:
        """Merge DB record (authoritative skeleton) with broadcaster snapshot (live progress)."""
        db_job = await video_job_manager.get_job(job_id)
        if db_job is None:
            return None

        snap = progress_broadcaster.snapshot(job_id) or {}
        status = VideoJobStatus(snap.get("status", db_job.status.value))
        default_progress = 100 if db_job.status == VideoJobStatus.COMPLETED else 0
        return VideoSwapStatusResponse(
            job_id=db_job.job_id,
            owner_key=db_job.owner_key,
            status=status,
            progress_percentage=snap.get("progress_percentage", default_progress),
            current_step=snap.get("current_step", db_job.status.value),
            message=snap.get("message", ""),
            processed_frames=snap.get("processed_frames", 0),
            total_frames=snap.get("total_frames", db_job.total_frames),
            output_url=db_job.output_url,
            error=db_job.error,
            created_at=db_job.created_at,
            updated_at=db_job.updated_at,
        )

    async def _publish(
        self,
        job_id: str,
        callback_url: Optional[str] = None,
        *,
        status: VideoJobStatus,
        progress_percentage: int,
        current_step: str,
        message: str,
        **extra,
    ) -> None:
        """Publish a progress update to WebSocket subscribers and, if configured,
        schedule a webhook POST to `callback_url`. Safe for per-frame updates as
        long as `callback_url` is None (we don't want to spam the receiver)."""
        payload = await progress_broadcaster.publish(
            job_id,
            status=status.value,
            progress_percentage=progress_percentage,
            current_step=current_step,
            message=message,
            **extra,
        )
        if callback_url:
            webhook_delivery.schedule(
                callback_url,
                {"event": _webhook_event(status), **payload},
            )

    async def _process_in_background(
        self,
        job_id: str,
        owner_key: str,
        image_url: str,
        video_url: str,
        output_path: str,
        temp_dir: str,
        callback_url: Optional[str] = None,
    ) -> None:
        source_image_path = os.path.join(temp_dir, "source_face.jpg")
        video_path = os.path.join(temp_dir, "input_video.mp4")
        loop = asyncio.get_running_loop()
        try:
            logger.info(f"[VideoSwap:{job_id}] Starting background processing for owner: {owner_key}")

            await video_job_manager.update_job(job_id, status=VideoJobStatus.DOWNLOADING)
            await self._publish(
                job_id,
                callback_url,
                status=VideoJobStatus.DOWNLOADING,
                progress_percentage=10,
                current_step="downloading_source_image",
                message="Downloading source image.",
            )
            await download_file(image_url, source_image_path)

            await self._publish(
                job_id,
                callback_url,
                status=VideoJobStatus.DOWNLOADING,
                progress_percentage=25,
                current_step="downloading_video",
                message="Downloading input video.",
            )
            await download_file(video_url, video_path)

            await self._publish(
                job_id,
                callback_url,
                status=VideoJobStatus.QUEUED,
                progress_percentage=30,
                current_step="waiting_for_processing_slot",
                message="Waiting for an available video processing slot.",
            )

            async with self._processing_semaphore:
                await video_job_manager.update_job(job_id, status=VideoJobStatus.PROCESSING)
                await self._publish(
                    job_id,
                    callback_url,
                    status=VideoJobStatus.PROCESSING,
                    progress_percentage=40,
                    current_step="processing_video",
                    message="Processing video frames.",
                )
                logger.info(f"[VideoSwap:{job_id}] Acquired processing slot")

                def frame_progress(processed_frames: int, total_frames: int) -> None:
                    progress = 40 + int((processed_frames / total_frames) * 50) if total_frames else 40
                    progress_broadcaster.publish_threadsafe(
                        loop,
                        job_id,
                        status=VideoJobStatus.PROCESSING.value,
                        progress_percentage=min(progress, 90),
                        current_step="processing_video",
                        message=f"Processing video frames: {processed_frames}/{total_frames or '?'}",
                        processed_frames=processed_frames,
                        total_frames=total_frames or None,
                    )

                await loop.run_in_executor(
                    self._executor,
                    process_video_face_swap,
                    source_image_path,
                    video_path,
                    output_path,
                    False,
                    frame_progress,
                )

                snap = progress_broadcaster.snapshot(job_id) or {}
                final_total_frames = snap.get("total_frames")
                if final_total_frames is not None:
                    await video_job_manager.update_job(job_id, total_frames=final_total_frames)

            output_filename = os.path.basename(output_path)
            final_output_url = build_local_output_url(output_filename)

            if settings.AWS_ACCESS_KEY_ID and settings.AWS_SECRET_ACCESS_KEY:
                s3_key = f"swapped_result/{owner_key}/{output_filename}"
                await video_job_manager.update_job(job_id, status=VideoJobStatus.UPLOADING)
                await self._publish(
                    job_id,
                    callback_url,
                    status=VideoJobStatus.UPLOADING,
                    progress_percentage=95,
                    current_step="uploading_result",
                    message="Uploading processed video.",
                )
                s3_url = await loop.run_in_executor(
                    self._executor, upload_to_s3, output_path, s3_key
                )

                if s3_url:
                    logger.info(f"[VideoSwap:{job_id}] Uploaded to S3: {s3_url}")
                    final_output_url = s3_url
                    try:
                        os.remove(output_path)
                    except Exception as e:
                        logger.warning(f"[VideoSwap:{job_id}] Failed to remove local file: {e}")
                else:
                    logger.info(f"[VideoSwap:{job_id}] S3 upload failed, falling back to local output: {output_path}")
            else:
                logger.info(f"[VideoSwap:{job_id}] Saved locally: {output_path}")

            await video_job_manager.mark_completed(job_id, output_url=final_output_url)
            await self._publish(
                job_id,
                callback_url,
                status=VideoJobStatus.COMPLETED,
                progress_percentage=100,
                current_step="completed",
                message="Video swap completed.",
                output_url=final_output_url,
            )

            try:
                os.remove(source_image_path)
                os.remove(video_path)
                os.rmdir(temp_dir)
                logger.info(f"[VideoSwap:{job_id}] Cleaned up temp files")
            except Exception as e:
                logger.warning(f"[VideoSwap:{job_id}] Failed to cleanup temp files: {e}")

            logger.info(f"[VideoSwap:{job_id}] Background processing completed successfully")

        except Exception as e:
            logger.error(f"[VideoSwap:{job_id}] Background processing failed: {e}")
            await video_job_manager.mark_failed(job_id, str(e))
            await self._publish(
                job_id,
                callback_url,
                status=VideoJobStatus.FAILED,
                progress_percentage=(progress_broadcaster.snapshot(job_id) or {}).get("progress_percentage", 0),
                current_step="failed",
                message="Video swap failed.",
                error=str(e),
            )
            try:
                for path in [source_image_path, video_path, output_path]:
                    if os.path.exists(path):
                        os.remove(path)
                if os.path.exists(temp_dir):
                    os.rmdir(temp_dir)
            except Exception as cleanup_err:
                logger.error(f"[VideoSwap:{job_id}] Cleanup after error failed: {cleanup_err}")


video_swap_service = VideoSwapService()
