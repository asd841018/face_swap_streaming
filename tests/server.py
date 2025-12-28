import cv2
import time
import threading
from typing import Optional
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
from app.model import RealTimeSwapper

RTSP_URL = "rtsp://127.0.0.1:8554/cam"
TARGET_SIZE = (1280, 720)

# --- Model init ---
providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
SWAPPER = RealTimeSwapper(
    providers=providers,
    face_analysis_name='buffalo_l',
    inswapper_path='models/inswapper_128.onnx'
)

# Prepare source face
source_img = cv2.imread('source/rose.jpeg')
src_faces = SWAPPER.get_source_face(source_img)

# Optional: try TurboJPEG for faster JPEG encoding
try:
    from turbojpeg import TurboJPEG
    _jpeg = TurboJPEG()
    USE_TURBOJPEG = True
except Exception:
    _jpeg = None
    USE_TURBOJPEG = False

JPEG_QUALITY = 80  # trade-off latency vs quality

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"],
)

# Shared latest frame bytes (JPEG) + version for change detection
_latest_lock = threading.Lock()
_latest_jpeg: Optional[bytes] = None
_latest_ver: int = 0
_frame_event = threading.Event()
_stop_flag = False

def encode_jpeg(img_bgr):
    if USE_TURBOJPEG:
        # TurboJPEG expects BGR=>RGB conversion
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return _jpeg.encode(img_rgb, quality=JPEG_QUALITY)
    else:
        ok, buf = cv2.imencode(".jpg", img_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), JPEG_QUALITY])
        if not ok:
            return None
        return buf.tobytes()

def producer_loop():
    global _latest_jpeg, _latest_ver, _stop_flag
    print("Connecting to RTSP...")
    cap = cv2.VideoCapture(RTSP_URL, cv2.CAP_FFMPEG)

    # Reduce internal buffering if your OpenCV/FFmpeg build supports it
    # (Some builds ignore this, but it doesn't hurt to try.)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    if not cap.isOpened():
        print("Error: Unable to open RTSP stream")
        return
    print("Connected. Starting low-latency processing.")

    # Strategy: read newest frames aggressively, drop stale ones
    while not _stop_flag:
        # Drain one frame to clear buffer (drop oldest)
        cap.read()

        ret, frame = cap.read()
        if not ret:
            print("Stream ended or error.")
            break

        # Start timer
        t0 = time.time()

        # Face swap (operate on a copy)
        swapped = SWAPPER.swap_into(frame, src_faces, swap_all=False)

        # Resize for display / bandwidth control
        swapped = cv2.resize(swapped, TARGET_SIZE)

        # Encode JPEG
        jpg = encode_jpeg(swapped)
        if jpg is None:
            continue

        # Publish latest (overwrite old; consumers only take newest)
        with _latest_lock:
            _latest_jpeg = jpg
            _latest_ver += 1
            ver = _latest_ver

        _frame_event.set()   # notify consumers a new frame is ready
        _frame_event.clear()

        # Optional: print instantaneous FPS (pipeline processing rate)
        dt = time.time() - t0
        if dt > 0:
            print(f"Pipeline FPS: {1.0/dt:.1f} (ver {ver})")

    cap.release()
    print("Producer stopped.")

@app.on_event("startup")
def _startup():
    th = threading.Thread(target=producer_loop, daemon=True)
    th.start()

@app.on_event("shutdown")
def _shutdown():
    global _stop_flag
    _stop_flag = True
    _frame_event.set()

INDEX_HTML = """
<!DOCTYPE html>
<html>
  <head>
    <meta charset="utf-8" />
    <title>Low-Latency AI Stream</title>
    <style>
      body { margin: 0; background: #0b0b0b; color: #eee; font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif; }
      header { padding: 12px 16px; font-weight: 600; }
      canvas { display: block; margin: 0 auto; width: 100vw; height: 56.25vw; max-width: 1280px; max-height: 720px; background: #111; }
      .bar { display:flex; align-items:center; gap:12px; padding: 8px 16px; font-size: 14px; color: #aaa;}
      .pill { padding: 2px 8px; border: 1px solid #333; border-radius: 999px;}
    </style>
  </head>
  <body>
    <header>AI Low-Latency Video</header>
    <div class="bar">
      <span class="pill" id="status">Connecting...</span>
      <span class="pill">Codec: JPEG</span>
      <span class="pill">Transport: WebSocket (binary)</span>
      <span class="pill" id="lat">Latency: - ms</span>
      <span class="pill" id="fps">Client FPS: -</span>
    </div>
    <canvas id="cv" width="1280" height="720"></canvas>

    <script>
      const canvas = document.getElementById('cv');
      const ctx = canvas.getContext('2d');
      const statusEl = document.getElementById('status');
      const latEl = document.getElementById('lat');
      const fpsEl = document.getElementById('fps');

      let lastFrameTime = performance.now();
      let frameCount = 0, lastFpsUpdate = performance.now();

      function updateFps() {
        const now = performance.now();
        frameCount++;
        if (now - lastFpsUpdate >= 1000) {
          fpsEl.textContent = "Client FPS: " + frameCount.toFixed(0);
          frameCount = 0;
          lastFpsUpdate = now;
        }
      }

      function drawBlob(blob, sentTs) {
        // Use createImageBitmap for faster decode & paint
        createImageBitmap(blob).then(img => {
          ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
          const now = performance.now();
          latEl.textContent = "Latency: " + (now - sentTs).toFixed(1) + " ms";
          updateFps();
        }).catch(() => {});
      }

      let ws;
      function connect() {
        const url = (location.protocol === 'https:' ? 'wss://' : 'ws://') + location.host + '/ws';
        ws = new WebSocket(url);
        ws.binaryType = 'blob';

        ws.onopen = () => { statusEl.textContent = "Connected"; };
        ws.onclose = () => { statusEl.textContent = "Disconnected. Reconnecting..."; setTimeout(connect, 1000); };
        ws.onerror = () => { statusEl.textContent = "Error"; };

        ws.onmessage = (ev) => {
          // Server prepends an 8-byte (float64) timestamp header (ms since epoch or performance time)
          // For simplicity we send as text header + binary body separated by a small delimiter.
          // But since we send as a single binary frame, we instead embed a simple 16-byte header:
          //   [0..7]   = Float64 sent_ts_ms
          //   [8..15]  = Uint32 width, Uint32 height (optional; not used here)
          if (ev.data instanceof Blob) {
            ev.data.arrayBuffer().then(buf => {
              const view = new DataView(buf);
              const sentTs = view.getFloat64(0, true);
              const jpegBytes = buf.slice(16); // skip header
              drawBlob(new Blob([jpegBytes], {type: 'image/jpeg'}), sentTs);
            });
          }
        };
      }
      connect();
    </script>
  </body>
</html>
"""

@app.get("/")
async def index():
    return HTMLResponse(INDEX_HTML)

@app.websocket("/ws")
async def ws_stream(ws: WebSocket):
    await ws.accept()
    last_sent_ver = -1
    try:
        while True:
            # Wait until a new frame is available (or poll very fast)
            # Using event helps reduce busy-waiting
            import asyncio
            await asyncio.to_thread(_frame_event.wait)
            _frame_event.clear()

            with _latest_lock:
                ver = _latest_ver
                jpg = _latest_jpeg

            # Only send if we have a newer frame than the one we last sent
            if jpg is not None and ver != last_sent_ver:
                last_sent_ver = ver
                # Build a small 16-byte header: [float64 sent_time_ms][uint32 w][uint32 h]
                sent_ms = time.perf_counter() * 1000.0
                header = bytearray(16)
                import struct
                struct.pack_into("<dII", header, 0, sent_ms, TARGET_SIZE[0], TARGET_SIZE[1])
                await ws.send_bytes(bytes(header) + jpg)
            # Tiny yield to avoid spinning in case of spurious wakeups
            await ws.receive_text() if False else None  # placeholder no-op
    except WebSocketDisconnect:
        pass
    except Exception as e:
        print("WS error:", e)
    finally:
        try:
            await ws.close()
        except Exception:
            pass

if __name__ == "__main__":
    # FastAPI uses uvicorn; set workers=1 to avoid duplicating the producer thread
    uvicorn.run("server:app", host="0.0.0.0", port=5000, reload=False)
