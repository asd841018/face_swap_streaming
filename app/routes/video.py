"""
Video face swap API endpoints
"""
import os
import uuid
import tempfile
import asyncio
import httpx
import cv2
import boto3
from botocore.exceptions import ClientError
from concurrent.futures import ThreadPoolExecutor
from fastapi import APIRouter, HTTPException, BackgroundTasks
from typing import Callable, Optional

from app.core import logger
from app.config import settings
from app.models.swapper import RealTimeSwapper
from app.schemas.video import VideoJobStatus, VideoSwapRequest, VideoSwapResponse, VideoSwapStatusResponse
from app.services.video_job_service import video_job_manager


router = APIRouter(prefix="/api/video", tags=["Video"])

# Output directory for processed videos (local fallback)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(CURRENT_DIR, "swapped_videos")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize swapper (lazy loading)
_swapper: Optional[RealTimeSwapper] = None
_s3_client = None
_bucket_name = None

# Thread pool executor for background video tasks.
_executor = ThreadPoolExecutor(max_workers=settings.MAX_WORKERS)

# Limit how many jobs can enter the actual model-processing phase at once.
_processing_semaphore = asyncio.Semaphore(max(1, settings.MAX_CONCURRENT_VIDEO_SWAP_JOBS))


def get_swapper() -> RealTimeSwapper:
    """Get or initialize the face swapper"""
    global _swapper
    if _swapper is None:
        _swapper = RealTimeSwapper(
            providers=['CUDAExecutionProvider', 'CPUExecutionProvider'],
            face_analysis_name=settings.FACE_ANALYSIS_NAME,
            inswapper_path=settings.SWAPPING_MODEL_PATH,
        )
    return _swapper


def get_s3_client():
    """Initialize S3 client with AWS credentials from settings."""
    global _s3_client, _bucket_name
    if _s3_client is None and settings.AWS_ACCESS_KEY_ID and settings.AWS_SECRET_ACCESS_KEY:
        _s3_client = boto3.client(
            "s3",
            region_name=settings.AWS_REGION,
            aws_access_key_id=settings.AWS_ACCESS_KEY_ID,
            aws_secret_access_key=settings.AWS_SECRET_ACCESS_KEY,
        )
        _bucket_name = settings.S3_UPLOAD_BUCKET
    return _s3_client, _bucket_name


def upload_to_s3(file_path: str, s3_key: str) -> Optional[str]:
    """
    Upload a file to S3 and return the public URL.
    
    Args:
        file_path: Local path to the file
        s3_key: S3 object key (path in bucket)
        
    Returns:
        S3 URL if successful, None otherwise
    """
    s3_client, bucket_name = get_s3_client()
    if not s3_client or not bucket_name:
        logger.warning("[S3] S3 not configured, skipping upload")
        return None
    
    try:
        s3_client.upload_file(
            file_path,
            bucket_name, # gsiai-dev-leo
            s3_key,  # swapped_result/{request.owner_key}/{output_filename}
        )
        
        # Generate the S3 URL
        s3_url = f"https://{bucket_name}.s3.{settings.AWS_REGION}.amazonaws.com/{s3_key}"
        logger.info(f"[S3] Uploaded to: {s3_url}")
        return s3_url
        
    except ClientError as e:
        logger.error(f"[S3] Upload failed: {e}")
        return None


async def download_file(url: str, save_path: str) -> None:
    """Download a file from URL"""
    async with httpx.AsyncClient(timeout=120.0) as client:
        response = await client.get(url)
        response.raise_for_status()
        with open(save_path, 'wb') as f:
            f.write(response.content)


def build_local_output_url(output_filename: str) -> str:
    """Build the local output URL."""
    return f"/static/output/{output_filename}"


def build_output_url(owner_key: str, output_filename: str) -> str:
    """Build the public output URL based on storage configuration."""
    if settings.AWS_ACCESS_KEY_ID and settings.AWS_SECRET_ACCESS_KEY:
        s3_key = f"swapped_result/{owner_key}/{output_filename}"
        return f"https://{settings.S3_UPLOAD_BUCKET}.s3.{settings.AWS_REGION}.amazonaws.com/{s3_key}"
    return build_local_output_url(output_filename)


def process_video_face_swap(
    source_image_path: str,
    video_path: str,
    output_path: str,
    swap_all: bool = False,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> None:
    """
    Process video and swap faces
    
    Args:
        source_image_path: Path to source face image
        video_path: Path to input video
        output_path: Path to save output video
        swap_all: Whether to swap all faces or just the largest one
        progress_callback: Optional callback for frame-based progress updates
    """
    swapper = get_swapper()
    
    # Load source face
    source_img = cv2.imread(source_image_path)
    if source_img is None:
        raise RuntimeError(f"Failed to load source image: {source_image_path}")
    
    source_face = swapper.get_source_face(source_img)
    
    # Open input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_interval = max(1, total_frames // 100) if total_frames > 0 else 100
    
    logger.info(f"[VideoSwap] Processing video: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    try:
        if progress_callback:
            progress_callback(0, total_frames)
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Swap face in frame
            swapped_frame = swapper.swap_into(frame, source_face, swap_all=swap_all)
            out.write(swapped_frame)
            
            frame_count += 1
            if frame_count % 100 == 0:
                logger.info(f"[VideoSwap] Processed {frame_count}/{total_frames} frames")
            if progress_callback and (frame_count == total_frames or frame_count % progress_interval == 0):
                progress_callback(frame_count, total_frames)
    finally:
        cap.release()
        out.release()
    
    logger.info(f"[VideoSwap] Completed: {frame_count} frames processed, saved to {output_path}")


async def process_video_in_background(
    job_id: str,
    owner_key: str,
    image_url: str,
    video_url: str,
    output_path: str,
    temp_dir: str,
) -> None:
    """
    Background task to process video face swap.
    This runs in a separate thread to avoid blocking the API response.
    """
    source_image_path = os.path.join(temp_dir, "source_face.jpg")
    video_path = os.path.join(temp_dir, "input_video.mp4")
    try:
        logger.info(f"[VideoSwap:{job_id}] Starting background processing for owner: {owner_key}")

        video_job_manager.update_job(
            job_id,
            status=VideoJobStatus.DOWNLOADING,
            progress_percentage=10,
            current_step="downloading_source_image",
            message="Downloading source image.",
        )
        await download_file(image_url, source_image_path)

        video_job_manager.update_job(
            job_id,
            status=VideoJobStatus.DOWNLOADING,
            progress_percentage=25,
            current_step="downloading_video",
            message="Downloading input video.",
        )
        await download_file(video_url, video_path)

        video_job_manager.update_job(
            job_id,
            status=VideoJobStatus.QUEUED,
            progress_percentage=30,
            current_step="waiting_for_processing_slot",
            message="Waiting for an available video processing slot.",
        )

        async with _processing_semaphore:
            video_job_manager.update_job(
                job_id,
                status=VideoJobStatus.PROCESSING,
                progress_percentage=40,
                current_step="processing_video",
                message="Processing video frames.",
            )
            logger.info(f"[VideoSwap:{job_id}] Acquired processing slot")

            # Process video in thread pool to avoid blocking
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(
                _executor,
                process_video_face_swap,
                source_image_path,
                video_path,
                output_path,
                False,
                lambda processed_frames, total_frames: video_job_manager.update_processing_progress(
                    job_id,
                    processed_frames,
                    total_frames,
                ),
            )

        output_filename = os.path.basename(output_path)
        final_output_url = build_local_output_url(output_filename)

        # Upload to S3 if configured
        if settings.AWS_ACCESS_KEY_ID and settings.AWS_SECRET_ACCESS_KEY:
            s3_key = f"swapped_result/{owner_key}/{output_filename}"
            video_job_manager.update_job(
                job_id,
                status=VideoJobStatus.UPLOADING,
                progress_percentage=95,
                current_step="uploading_result",
                message="Uploading processed video.",
            )
            s3_url = upload_to_s3(output_path, s3_key)

            if s3_url:
                logger.info(f"[VideoSwap:{job_id}] Uploaded to S3: {s3_url}")
                final_output_url = s3_url
                # Remove local file after successful S3 upload
                try:
                    os.remove(output_path)
                except Exception as e:
                    logger.warning(f"[VideoSwap:{job_id}] Failed to remove local file: {e}")
            else:
                logger.info(f"[VideoSwap:{job_id}] S3 upload failed, falling back to local output: {output_path}")
        else:
            logger.info(f"[VideoSwap:{job_id}] Saved locally: {output_path}")

        video_job_manager.mark_completed(job_id, final_output_url)

        # Cleanup temp files
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
        video_job_manager.mark_failed(job_id, str(e))
        # Cleanup on error
        try:
            for path in [source_image_path, video_path, output_path]:
                if os.path.exists(path):
                    os.remove(path)
            if os.path.exists(temp_dir):
                os.rmdir(temp_dir)
        except Exception as cleanup_err:
            logger.error(f"[VideoSwap:{job_id}] Cleanup after error failed: {cleanup_err}")


@router.post("/swap", response_model=VideoSwapResponse)
async def swap_video_face(request: VideoSwapRequest, background_tasks: BackgroundTasks):
    """
    Swap face in a video using the provided source face image.
    Returns immediately with job ID while processing in background.
    
    - **owner_key**: Unique identifier for the owner (used in output filename)
    - **image_url**: URL of the source face image
    - **video_url**: URL of the video to process
    
    Returns immediately with job metadata. Video will be processed in background.
    """
    try:
        # Generate unique ID for this job
        job_id = str(uuid.uuid4())[:8]
        temp_dir = tempfile.mkdtemp(prefix=f"faceswap_{job_id}_")
        output_filename = f"{request.owner_key}_{job_id}_swapped.mp4"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        output_url = build_output_url(request.owner_key, output_filename)
        status_url = f"/api/video/swap/{job_id}/status"

        video_job_manager.create_job(
            job_id=job_id,
            owner_key=request.owner_key,
            output_url=output_url,
        )

        # Add background task to process video
        background_tasks.add_task(
            process_video_in_background,
            job_id,
            request.owner_key,
            request.image_url,
            request.video_url,
            output_path,
            temp_dir,
        )

        logger.info(f"[VideoSwap:{job_id}] Job queued, returning immediately")
        
        return VideoSwapResponse(
            success=True,
            message=f"Video swap job {job_id} queued successfully. Processing in background.",
            job_id=job_id,
            output_url=output_url,
            status_url=status_url,
            owner_key=request.owner_key
        )
        
    except Exception as e:
        logger.error(f"[VideoSwap] Failed to queue job: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to queue job: {str(e)}")


@router.get("/swap/{job_id}/status", response_model=VideoSwapStatusResponse)
async def get_video_swap_status(job_id: str):
    """Poll the progress of a background video face swap job."""
    job = video_job_manager.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Job {job_id} not found")
    return job.to_response()
