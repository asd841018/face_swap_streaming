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
from pydantic import BaseModel, HttpUrl
from typing import Optional

from app.core import logger
from app.config import settings
from app.models.swapper import RealTimeSwapper
from app.schemas.video import VideoSwapRequest, VideoSwapResponse


router = APIRouter(prefix="/api/video", tags=["Video"])

# Output directory for processed videos (local fallback)
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(CURRENT_DIR, "swapped_videos")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize swapper (lazy loading)
_swapper: Optional[RealTimeSwapper] = None
_s3_client = None
_bucket_name = None

# Thread pool executor for parallel video processing (max 3 concurrent jobs)
_executor = ThreadPoolExecutor(max_workers=5)


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
        # Upload file with public-read ACL
        # fix this fucntion
        # upload_file(file, bucket, key)
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


def process_video_face_swap(
    source_image_path: str,
    video_path: str,
    output_path: str,
    swap_all: bool = False
) -> None:
    """
    Process video and swap faces
    
    Args:
        source_image_path: Path to source face image
        video_path: Path to input video
        output_path: Path to save output video
        swap_all: Whether to swap all faces or just the largest one
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
    
    logger.info(f"[VideoSwap] Processing video: {width}x{height} @ {fps}fps, {total_frames} frames")
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    try:
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
    finally:
        cap.release()
        out.release()
    
    logger.info(f"[VideoSwap] Completed: {frame_count} frames processed, saved to {output_path}")


async def process_video_in_background(
    job_id: str,
    owner_key: str,
    source_image_path: str,
    video_path: str,
    output_path: str,
    temp_dir: str
) -> None:
    """
    Background task to process video face swap.
    This runs in a separate thread to avoid blocking the API response.
    """
    try:
        logger.info(f"[VideoSwap:{job_id}] Starting background processing for owner: {owner_key}")
        
        # Process video in thread pool to avoid blocking
        loop = asyncio.get_event_loop()
        await loop.run_in_executor(
            _executor,
            process_video_face_swap,
            source_image_path,
            video_path,
            output_path
        )
        
        # Upload to S3 if configured
        s3_key = f"swapped_result/{owner_key}/{os.path.basename(output_path)}"
        s3_url = upload_to_s3(output_path, s3_key)
        
        if s3_url:
            logger.info(f"[VideoSwap:{job_id}] Uploaded to S3: {s3_url}")
            # Remove local file after successful S3 upload
            try:
                os.remove(output_path)
            except Exception as e:
                logger.warning(f"[VideoSwap:{job_id}] Failed to remove local file: {e}")
        else:
            logger.info(f"[VideoSwap:{job_id}] Saved locally: {output_path}")
        
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
    
    Returns immediately with the expected output URL. Video will be processed in background.
    """
    try:
        # Generate unique ID for this job
        job_id = str(uuid.uuid4())[:8]
        
        temp_dir = tempfile.mkdtemp(prefix=f"faceswap_{job_id}_")
        
        # Download source image
        source_image_path = os.path.join(temp_dir, "source_face.jpg")
        logger.info(f"[VideoSwap:{job_id}] Downloading source image: {request.image_url}")
        await download_file(request.image_url, source_image_path)
        
        # Download video
        video_path = os.path.join(temp_dir, "input_video.mp4")
        logger.info(f"[VideoSwap:{job_id}] Downloading video: {request.video_url}")
        await download_file(request.video_url, video_path)
        
        # Output path (associated with owner_key)
        output_filename = f"{request.owner_key}_{job_id}_swapped.mp4"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        # Add background task to process video
        background_tasks.add_task(
            process_video_in_background,
            job_id,
            request.owner_key,
            source_image_path,
            video_path,
            output_path,
            temp_dir
        )
        
        # Determine output URL (S3 or local)
        if settings.AWS_ACCESS_KEY_ID and settings.AWS_SECRET_ACCESS_KEY:
            # Will be uploaded to S3
            s3_key = f"swapped_result/{request.owner_key}/{output_filename}"
            output_url = f"https://{settings.S3_UPLOAD_BUCKET}.s3.{settings.AWS_REGION}.amazonaws.com/{s3_key}"
        else:
            # Will be saved locally
            output_url = f"/static/output/{output_filename}"
        
        logger.info(f"[VideoSwap:{job_id}] Job queued, returning immediately")
        
        return VideoSwapResponse(
            success=True,
            message=f"Video swap job {job_id} queued successfully. Processing in background.",
            output_url=output_url,
            owner_key=request.owner_key
        )
        
    except httpx.HTTPError as e:
        logger.error(f"[VideoSwap] Failed to download file: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to download file: {str(e)}")
    except Exception as e:
        logger.error(f"[VideoSwap] Failed to queue job: {e}")
        raise HTTPException(status_code=500, detail=f"Failed to queue job: {str(e)}")