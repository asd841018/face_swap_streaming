import cv2
import time
import subprocess as sp
import os
import urllib.request
import numpy as np
from app.models import RealTimeSwapper
from app.utils.frame_reader import FrameReader
from app.core import logger, settings

def load_source_face(swapper, source_face_url):
    source_img = None
    if source_face_url:
        try:
            logger.info(f"[ProcessWorker] Downloading source face from {source_face_url}...")
            # Set a timeout for the request
            with urllib.request.urlopen(source_face_url, timeout=10) as req:
                arr = np.asarray(bytearray(req.read()), dtype=np.uint8)
                source_img = cv2.imdecode(arr, -1)
            if source_img is None:
                logger.error(f"[ProcessWorker] Failed to decode image from {source_face_url}")
        except Exception as e:
            logger.error(f"[ProcessWorker] Failed to download source face: {e}")

    if source_img is None:
        # Fallback to default if download fails or no URL provided
        # Only if we really need a default. 
        # But if this is an update, maybe we shouldn't fallback to default but keep previous?
        # For now, let's return None if failed, so caller can decide.
        return None
        
    src_faces = swapper.get_source_face(source_img)
    return src_faces

def run_stream_process(stop_event, queue, input_rtmp, output_rtmp, source_face_url=None, video_config=None):
    """
    This function runs in a separate process.
    It initializes its own model instance to avoid GIL issues and maximize CPU usage.
    GPU usage will still be shared, but Python overhead is parallelized.
    
    Args:
        stop_event: multiprocessing.Event to signal stop
        queue: multiprocessing.Queue for receiving messages
        input_rtmp: Input RTMP URL
        output_rtmp: Output RTMP URL
        source_face_url: URL of source face image (optional)
        video_config: Dictionary with video settings (optional)
            - bitrate: e.g., "3000k"
            - resolution: e.g., "1280x720"
            - frame_rate: e.g., 25
            - swap_all: bool
    """
    logger.info(f"[ProcessWorker] Starting process for: {input_rtmp} -> {output_rtmp}")
    
    # Parse video config
    if video_config is None:
        video_config = {}
    
    video_bitrate = video_config.get("bitrate", "3000k")
    video_resolution = video_config.get("resolution", "1280x720")
    frame_rate = video_config.get("frame_rate", 30)
    
    # Parse resolution
    try:
        W_OUT, H_OUT = map(int, video_resolution.split('x'))
    except:
        W_OUT, H_OUT = 1280, 720
    
    # Initialize Model inside the process
    # Note: Each process will load its own model into VRAM.
    # If VRAM is limited, this might cause OOM.
    # However, for 5 streams, it might be better to share model weights if possible,
    # but ONNX Runtime usually handles this well or we rely on OS paging if needed.
    # Ideally, we want to share the model, but Python multiprocessing makes sharing complex objects hard.
    # A better approach for VRAM efficiency with multiprocessing is to have a single Model Process 
    # that receives frames from multiple Reader Processes via Queue, processes them, and sends back.
    # BUT, for simplicity and to avoid GIL, let's try independent processes first.
    
    try:
        # Hardcoded paths for now, similar to previous worker.py
        # Adjust paths as needed based on where this is run
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
        # model_path = os.path.join(root_dir, '.assets/models/dynamic_batch_model.onnx')
        
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        swapper = RealTimeSwapper(
            providers=providers,
            face_analysis_name=settings.FACE_ANALYSIS_NAME,
            inswapper_path=settings.SWAPPING_MODEL_PATH
        )
        
        src_faces = load_source_face(swapper, source_face_url)
        
        if src_faces is None:
            source_img_path = os.path.join(root_dir, '.assets/source/rose.jpeg')
            logger.info(f"[ProcessWorker] Using default source face: {source_img_path}")
            source_img = cv2.imread(source_img_path)
            src_faces = swapper.get_source_face(source_img)

        logger.info("[ProcessWorker] Model loaded.")
        
    except Exception as e:
        logger.error(f"[ProcessWorker] Failed to initialize model: {e}")
        return

    # FFmpeg command - use configured values
    # command = [
    #     'ffmpeg',
    #     '-y',
    #     '-f', 'rawvideo',
    #     '-vcodec', 'rawvideo',
    #     '-pix_fmt', 'bgr24',
    #     '-s', f"{W_OUT}x{H_OUT}",
    #     '-r', str(frame_rate),
    #     '-i', '-',
    #     '-c:v', 'h264_nvenc',
    #     '-preset', 'p4',
    #     '-tune', 'll',
    #     '-rc', 'cbr',
    #     '-b:v', video_bitrate,
    #     '-maxrate', video_bitrate,
    #     '-bufsize', str(int(video_bitrate[:-1]) * 2) + video_bitrate[-1],  # 通常設置1.25-2倍的bitrate作為bufsize
    #     '-g', str(frame_rate * 2),  # keyframe interval = 2 seconds
    #     '-bf', '0',
    #     '-pix_fmt', 'yuv420p',
    #     '-an',
    #     '-f', 'flv',
    #     '-flvflags', 'no_duration_filesize',
    #     output_rtmp
    # ]
    
    command = [
        'ffmpeg',
        '-y',
        '-f', 'rawvideo',
        '-vcodec', 'rawvideo',
        '-pix_fmt', 'bgr24',
        '-s', f"{W_OUT}x{H_OUT}",
        '-r', str(frame_rate),
        '-i', '-',
        '-c:v', 'h264_nvenc',
    
        # --- 修改點 1: 犧牲一點點畫質，換取最快速度 (解決 0.6x 卡頓問題) ---
        '-preset', 'p1',  # 改用 p1 (最快) 或 p2。原來的 p4 運算量太大，導致你推流來不及。
        
        # --- 修改點 2: 使用「超」低延遲模式 ---
        '-tune', 'ull',   # 改用 ull (Ultra Low Latency)。原來的 ll 還不夠激進。
        '-rc', 'cbr',
        '-b:v', video_bitrate,
        '-maxrate', video_bitrate,
        '-bufsize', str(int(video_bitrate[:-1]) * 2) + video_bitrate[-1],
        '-g', str(frame_rate * 2),
        '-bf', '0',

        # --- 修改點 3: 強制關閉預讀 (關鍵！解決 reordered frames) ---
        '-rc-lookahead', '0', # 叫顯卡不要偷看後面的畫面，直接編碼送出
        '-pix_fmt', 'yuv420p',
        '-an',
        '-f', 'flv',
        '-flvflags', 'no_duration_filesize',
        output_rtmp
    ]

    ffmpeg_process = None
    reader = None

    try:
        ffmpeg_process = sp.Popen(command, stdin=sp.PIPE)
    except FileNotFoundError:
        logger.error("[ProcessWorker] Error: FFmpeg not found")
        return

    # Retry connecting to the stream for a few seconds
    max_retries = 10
    for i in range(max_retries):
        reader = FrameReader(input_rtmp)
        if reader.connected:
            break
        logger.info(f"[ProcessWorker] Waiting for stream... ({i+1}/{max_retries})")
        time.sleep(1)
    
    if not reader or not reader.connected:
        logger.error("[ProcessWorker] Error: Could not open input stream after retries")
        if ffmpeg_process.stdin: ffmpeg_process.stdin.close()
        ffmpeg_process.wait()
        return
    reader.start()

    frame_count = 0
    start_time = time.time()
    last_frame_time = 0
    target_interval = 1.0 / frame_rate

    exit_reason = "unknown"
    try:
        while not stop_event.is_set() and reader.running:
            # Check for updates from the main process
            if not queue.empty():
                try:
                    message = queue.get_nowait()
                    if message.get('type') == 'update_source_face':
                        new_url = message.get('url')
                        logger.info(f"[ProcessWorker] Received update source face request: {new_url}")
                        new_src_faces = load_source_face(swapper, new_url)
                        if new_src_faces is not None:
                            src_faces = new_src_faces
                            logger.info("[ProcessWorker] Source face updated successfully.")
                        else:
                            logger.warning("[ProcessWorker] Failed to update source face, keeping previous one.")
                except Exception as e:
                    logger.error(f"[ProcessWorker] Error processing queue message: {e}")

            # Throttle to target frame rate
            now = time.time()
            if now - last_frame_time < target_interval:
                time.sleep(0.001)
                continue
            
            frame = reader.get_latest_frame()
            if frame is None:
                time.sleep(0.01)
                continue

            last_frame_time = time.time()

            try:
                # Resize frame to match FFmpeg expected resolution
                # if frame.shape[1] != W_OUT or frame.shape[0] != H_OUT:
                    # frame = cv2.resize(frame, (W_OUT, H_OUT))
                
                # swapped_frame = swapper.swap_into(frame, src_faces, swap_all=swap_all)
                swapped_frame = swapper.deform_face(frame)
                if swapped_frame.shape[1] != W_OUT or swapped_frame.shape[0] != H_OUT:
                    swapped_frame = cv2.resize(swapped_frame, (W_OUT, H_OUT))

                ffmpeg_process.stdin.write(swapped_frame.tobytes())
                # ffmpeg_process.stdin.write(frame.tobytes())
                
                frame_count += 1
                if frame_count % 100 == 0:
                    elapsed = time.time() - start_time
                    logger.info(f"[ProcessWorker] PID {os.getpid()} FPS: {frame_count / elapsed:.2f}")

            except Exception as e:
                logger.error(f"[ProcessWorker] Processing error: {e}")
                exit_reason = f"processing_error: {e}"
                break
        
        # Determine why the loop exited
        if stop_event.is_set():
            exit_reason = "stop_event_received"
        elif not reader.running:
            exit_reason = "input_stream_disconnected"
        else:
            exit_reason = "normal_exit"
            
    except Exception as e:
        logger.error(f"[ProcessWorker] Unexpected error: {e}")
        exit_reason = f"unexpected_error: {e}"
    finally:
        logger.info(f"[ProcessWorker] PID {os.getpid()} Cleaning up... (reason: {exit_reason})")
        if reader:
            reader.stop()
            reader.join()
        if ffmpeg_process and ffmpeg_process.stdin:
            ffmpeg_process.stdin.close()
        if ffmpeg_process:
            ffmpeg_process.wait()
        logger.info(f"[ProcessWorker] PID {os.getpid()} Stopped")

