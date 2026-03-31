import cv2
import time
import subprocess as sp
import os
import urllib.request
import numpy as np
import threading
from dataclasses import dataclass
from app.models import RealTimeSwapper
from app.utils.frame_reader import FrameReader
from app.core import logger
from app.config import settings


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

@dataclass
class VideoConfig:
    """Parsed video output settings."""
    bitrate: str
    width: int
    height: int
    fps: int
    swap_all: bool


# ---------------------------------------------------------------------------
# Config parsing
# ---------------------------------------------------------------------------

def _parse_video_config(raw: dict | None) -> VideoConfig:
    raw = raw or {}
    bitrate = raw.get("bitrate") or settings.DEFAULT_VIDEO_BITRATE
    resolution = raw.get("resolution") or settings.DEFAULT_VIDEO_RESOLUTION
    swap_all = bool(raw.get("swap_all", False))
    try:
        fps = max(1, int(raw.get("frame_rate") or settings.DEFAULT_FRAME_RATE))
    except (TypeError, ValueError):
        fps = settings.DEFAULT_FRAME_RATE
    try:
        w, h = map(int, resolution.split("x"))
    except Exception:
        w, h = 1280, 720
    return VideoConfig(bitrate=bitrate, width=w, height=h, fps=fps, swap_all=swap_all)


def _bufsize(bitrate: str) -> str:
    b = (bitrate or "").strip().lower()
    if not b:
        return "6000k"
    suffix = b[-1]
    val_str, unit = (b[:-1], suffix) if suffix.isalpha() else (b, "k")
    try:
        return f"{int(val_str) * 2}{unit}"
    except ValueError:
        return "6000k"


# ---------------------------------------------------------------------------
# Face loading
# ---------------------------------------------------------------------------

def _default_source_face_path() -> str:
    configured = os.path.join(settings.SOURCE_FACE_DIR or "", settings.SOURCE_FACE or "rose.jpeg")
    if os.path.exists(configured):
        return configured
    root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    return os.path.join(root, ".assets/source/rose.jpeg")


def load_source_face(swapper: RealTimeSwapper, url: str | None):
    """Download image from *url* and return face embedding, or None."""
    if not url:
        return None
    try:
        with urllib.request.urlopen(url, timeout=10) as req:
            arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            logger.error(f"[Worker] Cannot decode image: {url}")
            return None
        return swapper.get_source_face(img)
    except Exception as e:
        logger.error(f"[Worker] Download source face failed: {e}")
        return None


def _load_face_or_default(swapper: RealTimeSwapper, url: str | None):
    """Try loading face from *url*; fall back to the default local image."""
    face = load_source_face(swapper, url)
    if face is not None:
        return face
    path = _default_source_face_path()
    logger.info(f"[Worker] Using default source face: {path}")
    img = cv2.imread(path)
    if img is not None:
        return swapper.get_source_face(img)
    return None


# ---------------------------------------------------------------------------
# Queue message handling
# ---------------------------------------------------------------------------

def _handle_kol_update(msg: dict, mtype: str, swapper: RealTimeSwapper, state: dict):
    """Handle update_kol_mode / update_kol_source_url queue messages."""
    if mtype == "update_kol_mode":
        state["is_kol_mode"] = bool(msg.get("is_kol_mode", False))
        state["kol_source_url"] = msg.get("kol_source_url")
    else:
        state["kol_source_url"] = msg.get("kol_source_url")

    if state["is_kol_mode"] and state["kol_source_url"]:
        new = load_source_face(swapper, state["kol_source_url"])
        if new is not None:
            state["src_faces"] = new
            logger.info("[Worker] Switched to KOL source face.")
    elif not state["is_kol_mode"]:
        new = _load_face_or_default(swapper, state.get("source_face_url"))
        if new is not None:
            state["src_faces"] = new
        logger.info("[Worker] KOL mode off, reverted to original source face.")


def _process_queue_messages(queue, swapper: RealTimeSwapper, state: dict):
    """Drain pending messages and mutate *state* in-place."""
    while not queue.empty():
        try:
            msg = queue.get_nowait()
        except Exception:
            break
        mtype = msg.get("type")
        if mtype == "update_source_face":
            new = load_source_face(swapper, msg.get("url"))
            if new is not None:
                state["src_faces"] = new
                logger.info("[Worker] Source face updated.")
        elif mtype == "update_use_image_filter":
            state["use_image_filter"] = bool(msg.get("use_image_filter", False))
        elif mtype == "update_filter_type":
            state["filter_type"] = msg.get("filter_type")
        elif mtype in ("update_kol_mode", "update_kol_source_url"):
            _handle_kol_update(msg, mtype, swapper, state)


# ---------------------------------------------------------------------------
# FFmpeg helpers
# ---------------------------------------------------------------------------

def _build_ffmpeg_cmd(w: int, h: int, fps: int, bitrate: str, output: str) -> list[str]:
    return [
        "ffmpeg", "-y",
        "-f", "rawvideo", "-vcodec", "rawvideo", "-pix_fmt", "bgr24",
        "-s", f"{w}x{h}", "-r", str(fps), "-i", "-",
        "-c:v", "h264_nvenc", "-preset", "p1", "-tune", "ull",
        "-rc", "cbr",
        "-b:v", bitrate, "-maxrate", bitrate, "-bufsize", _bufsize(bitrate),
        "-g", str(fps * 2), "-bf", "0", "-rc-lookahead", "0",
        "-pix_fmt", "yuv420p", "-an",
        "-f", "flv", "-flvflags", "no_duration_filesize",
        output,
    ]


def _drain_ffmpeg_stderr(proc: sp.Popen, stop: threading.Event):
    if not proc or not proc.stderr:
        return
    while not stop.is_set():
        try:
            raw = proc.stderr.readline()
            if not raw:
                break
            line = raw.decode("utf-8", errors="ignore").strip()
            if not line:
                continue
            low = line.lower()
            if any(w in low for w in ("error", "failed", "invalid", "fatal")):
                logger.error(f"[Worker][FFmpeg] {line}")
            elif "warning" in low:
                logger.warning(f"[Worker][FFmpeg] {line}")
        except Exception:
            break


def _start_ffmpeg(
    vcfg: VideoConfig, output_rtmp: str,
) -> tuple[sp.Popen, threading.Event, threading.Thread]:
    """Launch FFmpeg and a daemon thread draining its stderr."""
    stderr_stop = threading.Event()
    ffmpeg = sp.Popen(
        _build_ffmpeg_cmd(vcfg.width, vcfg.height, vcfg.fps, vcfg.bitrate, output_rtmp),
        stdin=sp.PIPE, stderr=sp.PIPE,
    )
    thread = threading.Thread(
        target=_drain_ffmpeg_stderr, args=(ffmpeg, stderr_stop), daemon=True,
    )
    thread.start()
    return ffmpeg, stderr_stop, thread


def _cleanup(reader, ffmpeg, stderr_stop, stderr_thread, reason):
    pid = os.getpid()
    logger.info(f"[Worker] PID {pid} cleaning up (reason: {reason})")
    stderr_stop.set()

    if reader:
        reader.stop()
        reader.join(timeout=2)

    if ffmpeg and ffmpeg.stdin and not ffmpeg.stdin.closed:
        try:
            ffmpeg.stdin.close()
        except Exception:
            pass

    if ffmpeg:
        try:
            ffmpeg.wait(timeout=5)
        except sp.TimeoutExpired:
            ffmpeg.terminate()
            try:
                ffmpeg.wait(timeout=2)
            except sp.TimeoutExpired:
                ffmpeg.kill()

    if ffmpeg and ffmpeg.stderr and not ffmpeg.stderr.closed:
        try:
            ffmpeg.stderr.close()
        except Exception:
            pass

    if stderr_thread and stderr_thread.is_alive():
        stderr_thread.join(timeout=1)

    logger.info(f"[Worker] PID {pid} stopped.")


# ---------------------------------------------------------------------------
# Startup helpers
# ---------------------------------------------------------------------------

def _init_swapper(
    source_face_url: str | None,
    is_kol_mode: bool,
    kol_source_url: str | None,
) -> tuple[RealTimeSwapper, object]:
    """Create the AI model and load the initial source face.

    Returns ``(swapper, src_faces)``.  Raises on model-init failure.
    """
    swapper = RealTimeSwapper(
        providers=["CUDAExecutionProvider"],
        face_analysis_name=settings.FACE_ANALYSIS_NAME,
        inswapper_path=settings.SWAPPING_MODEL_PATH,
    )
    if is_kol_mode and kol_source_url:
        logger.info(f"[Worker] KOL mode enabled. Loading KOL source face from {kol_source_url}")
        url = kol_source_url
    else:
        url = source_face_url
    src_faces = _load_face_or_default(swapper, url)
    logger.info("[Worker] Model loaded.")
    return swapper, src_faces


def _connect_input_stream(input_rtmp: str, max_attempts: int = 10) -> FrameReader | None:
    """Try to open the input RTMP stream, retrying up to *max_attempts* times."""
    for attempt in range(1, max_attempts + 1):
        reader = FrameReader(input_rtmp)
        if reader.connected:
            return reader
        logger.info(f"[Worker] Waiting for stream... ({attempt}/{max_attempts})")
        time.sleep(1)
    return None


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def _ensure_resolution(frame: np.ndarray, w: int, h: int) -> np.ndarray:
    if frame.shape[1] != w or frame.shape[0] != h:
        return cv2.resize(frame, (w, h))
    return frame


def _run_main_loop(stop_event, queue, reader, ffmpeg, swapper, state, vcfg):
    """Run the frame-processing loop.  Returns the exit-reason string."""
    frame_count = 0
    t0 = time.time()
    last_frame_t = 0.0
    interval = 1.0 / vcfg.fps
    exit_reason = "unknown"

    try:
        while not stop_event.is_set() and reader.running:
            if ffmpeg.poll() is not None:
                exit_reason = f"ffmpeg_exited_{ffmpeg.returncode}"
                break

            _process_queue_messages(queue, swapper, state)

            now = time.time()
            if now - last_frame_t < interval:
                time.sleep(0.001)
                continue

            frame = reader.get_latest_frame()
            if frame is None:
                time.sleep(0.01)
                continue
            last_frame_t = time.time()

            try:
                frame = _ensure_resolution(frame, vcfg.width, vcfg.height)

                if state["use_image_filter"]:
                    out = swapper.deform_face(frame, filter_type=state["filter_type"])
                else:
                    if state["src_faces"] is None:
                        continue
                    out = swapper.swap_into(frame, state["src_faces"], swap_all=vcfg.swap_all)

                out = _ensure_resolution(out, vcfg.width, vcfg.height)
                ffmpeg.stdin.write(out.tobytes())
                frame_count += 1
                if frame_count % 200 == 0:
                    logger.info(f"[Worker] PID {os.getpid()} FPS: {frame_count / (time.time() - t0):.1f}")
            except BrokenPipeError:
                exit_reason = "ffmpeg_broken_pipe"
                break
            except Exception as e:
                logger.error(f"[Worker] Processing error: {e}")
                exit_reason = "processing_error"
                break

        if exit_reason == "unknown":
            exit_reason = (
                "stop_event" if stop_event.is_set()
                else "stream_disconnected" if not reader.running
                else "normal_exit"
            )
    except Exception as e:
        logger.error(f"[Worker] Unexpected error: {e}")
        exit_reason = "unexpected_error"

    return exit_reason


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run_stream_process(
    stop_event,
    queue,
    input_rtmp: str,
    output_rtmp: str,
    source_face_url: str | None = None,
    use_image_filter: bool = False,
    filter_type: str | None = None,
    is_kol_mode: bool = False,
    kol_source_url: str | None = None,
    video_config: dict | None = None,
):
    """Worker process entry point: read RTMP -> AI -> push RTMP via FFmpeg."""
    logger.info(f"[Worker] Starting: {input_rtmp} -> {output_rtmp}")
    vcfg = _parse_video_config(video_config)

    try:
        swapper, src_faces = _init_swapper(source_face_url, is_kol_mode, kol_source_url)
    except Exception as e:
        logger.error(f"[Worker] Model init failed: {e}")
        return

    try:
        ffmpeg, stderr_stop, stderr_thread = _start_ffmpeg(vcfg, output_rtmp)
    except FileNotFoundError:
        logger.error("[Worker] FFmpeg not found.")
        return

    reader = _connect_input_stream(input_rtmp)
    if reader is None:
        logger.error("[Worker] Could not open input stream.")
        _cleanup(None, ffmpeg, stderr_stop, stderr_thread, "input_connection_failed")
        return
    reader.start()

    state = {
        "src_faces": src_faces,
        "source_face_url": source_face_url,
        "use_image_filter": use_image_filter,
        "filter_type": filter_type,
        "is_kol_mode": is_kol_mode,
        "kol_source_url": kol_source_url,
    }
    exit_reason = "unknown"
    try:
        exit_reason = _run_main_loop(stop_event, queue, reader, ffmpeg, swapper, state, vcfg)
    finally:
        _cleanup(reader, ffmpeg, stderr_stop, stderr_thread, exit_reason)
