"""Shared helpers for the video face-swap feature."""
from typing import Callable, Optional

import cv2
import httpx
import boto3
from botocore.exceptions import ClientError

from app.core import logger
from app.config import settings
from app.models.swapper import RealTimeSwapper


_swapper: Optional[RealTimeSwapper] = None
_s3_client = None
_bucket_name: Optional[str] = None


def get_swapper() -> RealTimeSwapper:
    """Get or initialize the face swapper (lazy singleton)."""
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
    """Upload a file to S3 and return the public URL (or None on failure)."""
    s3_client, bucket_name = get_s3_client()
    if not s3_client or not bucket_name:
        logger.warning("[S3] S3 not configured, skipping upload")
        return None

    try:
        s3_client.upload_file(file_path, bucket_name, s3_key)
        s3_url = f"https://{bucket_name}.s3.{settings.AWS_REGION}.amazonaws.com/{s3_key}"
        logger.info(f"[S3] Uploaded to: {s3_url}")
        return s3_url
    except ClientError as e:
        logger.error(f"[S3] Upload failed: {e}")
        return None


async def download_file(url: str, save_path: str) -> None:
    """Download a file from URL to the given local path."""
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
    """Process video and swap faces frame-by-frame."""
    swapper = get_swapper()

    source_img = cv2.imread(source_image_path)
    if source_img is None:
        raise RuntimeError(f"Failed to load source image: {source_image_path}")

    source_face = swapper.get_source_face(source_img)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    progress_interval = max(1, total_frames // 100) if total_frames > 0 else 100

    logger.info(f"[VideoSwap] Processing video: {width}x{height} @ {fps}fps, {total_frames} frames")

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
