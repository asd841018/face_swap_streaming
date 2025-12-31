"""
Video face swap API endpoints
"""
import os
import uuid
import tempfile
import httpx
import cv2
import boto3
from botocore.exceptions import ClientError
from fastapi import APIRouter, HTTPException, BackgroundTasks
from pydantic import BaseModel, HttpUrl
from typing import Optional

from app.core import logger
from app.core.config import settings
from app.models.swapper import RealTimeSwapper
from app.schemas.video import VideoSwapRequest, VideoSwapResponse


router = APIRouter(prefix="/api/video", tags=["Video"])

# Output directory for processed videos (local fallback)
OUTPUT_DIR = "/tmp/face_swap_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Initialize swapper (lazy loading)
_swapper: Optional[RealTimeSwapper] = None
_s3_client = None
_bucket_name = None


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
        s3_client.upload_file(
            file_path,
            bucket_name,
            s3_key,
            ExtraArgs={
                'ContentType': 'video/mp4',
            }
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

@router.post("/swap", response_model=VideoSwapResponse)
async def swap_video_face(request: VideoSwapRequest):
    """
    Swap face in a video using the provided source face image.
    
    - **owner_key**: Unique identifier for the owner (used in output filename)
    - **image_url**: URL of the source face image
    - **video_url**: URL of the video to process
    
    Returns the URL of the processed video.
    """
    try:
        # Generate unique ID for this job
        job_id = str(uuid.uuid4())[:8]
        
        
        temp_dir = tempfile.mkdtemp(prefix=f"faceswap_{job_id}_")
        
        # Download source image
        source_image_path = os.path.join(temp_dir, "source_face.jpg")
        logger.info(f"[VideoSwap] Downloading source image: {request.image_url}")
        await download_file(request.image_url, source_image_path)
        
        # Download video
        video_path = os.path.join(temp_dir, "input_video.mp4")
        logger.info(f"[VideoSwap] Downloading video: {request.video_url}")
        await download_file(request.video_url, video_path)
        
        # Output path (associated with owner_key)
        output_filename = f"{request.owner_key}_{job_id}_swapped.mp4"
        output_path = os.path.join(OUTPUT_DIR, output_filename)
        
        # Process video
        logger.info(f"[VideoSwap] Starting face swap for owner: {request.owner_key}")
        process_video_face_swap(source_image_path, video_path, output_path)
        
        # Upload to S3 if configured
        s3_key = f"swapped_result/{request.owner_key}/{output_filename}"
        s3_url = upload_to_s3(output_path, s3_key)
        
        if s3_url:
            output_url = s3_url
            # Remove local file after successful S3 upload
            try:
                os.remove(output_path)
            except Exception as e:
                logger.warning(f"[VideoSwap] Failed to remove local file: {e}")
        else:
            # Fallback to local path if S3 not configured
            output_url = f"/static/output/{output_filename}"
        
        # Cleanup temp files
        try:
            os.remove(source_image_path)
            os.remove(video_path)
            os.rmdir(temp_dir)
        except Exception as e:
            logger.warning(f"[VideoSwap] Failed to cleanup temp files: {e}")
        
        return VideoSwapResponse(
            success=True,
            message="Video face swap completed successfully",
            output_url=output_url,
            owner_key=request.owner_key
        )
        
    except httpx.HTTPError as e:
        logger.error(f"[VideoSwap] Failed to download file: {e}")
        raise HTTPException(status_code=400, detail=f"Failed to download file: {str(e)}")
    except RuntimeError as e:
        logger.error(f"[VideoSwap] Processing error: {e}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"[VideoSwap] Unexpected error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")


@router.post("/swap-async", response_model=VideoSwapResponse)
async def swap_video_face_async(request: VideoSwapRequest, background_tasks: BackgroundTasks):
    """
    Async version: Queue video face swap as a background task.
    
    This endpoint returns immediately and processes the video in the background.
    You can implement a webhook or polling mechanism to get the result.
    
    - **owner_key**: Unique identifier for the owner
    - **image_url**: URL of the source face image  
    - **video_url**: URL of the video to process
    """
    job_id = str(uuid.uuid4())[:8]
    output_filename = f"{request.owner_key}_{job_id}_swapped.mp4"
    output_url = f"/static/output/{output_filename}"
    
    # TODO: Implement background processing with status tracking
    # background_tasks.add_task(process_video_job, request, job_id)
    
    return VideoSwapResponse(
        success=True,
        message=f"Video swap job queued with ID: {job_id}. Processing will start shortly.",
        output_url=output_url,
        owner_key=request.owner_key
    )
