"""Video face-swap API endpoints."""
from fastapi import APIRouter, BackgroundTasks, HTTPException, WebSocket, WebSocketDisconnect

from app.core import logger
from app.schemas.video import (
    VideoJobStatus,
    VideoSwapRequest,
    VideoSwapResponse,
    VideoSwapStatusResponse,
)
from app.services.progress_broadcaster import progress_broadcaster
from app.services.video_job_service import video_job_manager
from app.video_swap.service import video_swap_service


router = APIRouter(prefix="/api/video", tags=["Video"])


@router.post("/swap", response_model=VideoSwapResponse)
async def swap_video_face(request: VideoSwapRequest, background_tasks: BackgroundTasks):
    """
    Swap face in a video using the provided source face image.
    Returns immediately with job ID while processing in background.

    - **owner_key**: Unique identifier for the owner (used in output filename)
    - **image_url**: URL of the source face image
    - **video_url**: URL of the video to process
    """
    try:
        return await video_swap_service.queue_swap_job(request, background_tasks)
    except Exception as e:
        logger.error(f"[VideoSwap] Failed to queue job: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to queue job: {str(e)}")


@router.get("/busy")
async def is_busy():
    """Return whether the server is still processing any video swap job.

    Used by the SaaS platform to decide whether this VM can be safely shut down.
    """
    return {"busy": await video_job_manager.has_active_jobs()}


@router.get("/swap/{job_id}/status", response_model=VideoSwapStatusResponse)
async def get_video_swap_status(job_id: str):
    """Poll the progress of a background video face swap job."""
    response = await video_swap_service.get_job_status(job_id)
    if response is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return response


_TERMINAL_STATUSES = {VideoJobStatus.COMPLETED.value, VideoJobStatus.FAILED.value}


@router.websocket("/swap/{job_id}/ws")
async def video_swap_progress_ws(websocket: WebSocket, job_id: str):
    """WebSocket stream of live progress for a video swap job.

    Connects, sends the current snapshot immediately, then pushes every update
    until the job reaches a terminal state (completed/failed) or the client
    disconnects.
    """
    db_job = await video_job_manager.get_job(job_id)
    if db_job is None:
        await websocket.close(code=4404, reason="job not found")
        return

    await websocket.accept()
    queue = await progress_broadcaster.subscribe(job_id)
    try:
        if db_job.status.value in _TERMINAL_STATUSES and progress_broadcaster.snapshot(job_id) is None:
            await websocket.send_json({
                "job_id": db_job.job_id,
                "owner_key": db_job.owner_key,
                "status": db_job.status.value,
                "progress_percentage": 100 if db_job.status == VideoJobStatus.COMPLETED else 0,
                "current_step": db_job.status.value,
                "message": "Job already finished.",
                "output_url": db_job.output_url,
                "error": db_job.error,
            })
            return

        while True:
            payload = await queue.get()
            await websocket.send_json(payload)
            if payload.get("status") in _TERMINAL_STATUSES:
                break
    except WebSocketDisconnect:
        pass
    except Exception as e:
        logger.error(f"[VideoSwap:{job_id}] WS error: {e}")
    finally:
        await progress_broadcaster.unsubscribe(job_id, queue)
        try:
            await websocket.close()
        except Exception:
            pass
