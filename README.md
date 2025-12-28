# This is FOR my backend and multiprocessing practice

A real-time RTMP stream processing service based on FastAPI and MediaMTX. It receives live streams from users, performs real-time AI face swapping/filtering, and pushes the processed video to a specified destination (such as a local MediaMTX server or external platforms like YouTube/Twitch).

## Architecture

*   **MediaMTX**: Responsible for receiving RTMP streams and notifying the backend via Webhooks.
*   **FastAPI Backend**: Receives Webhook events and manages Worker Processes.
*   **Worker Process**: An independent process responsible for:
    1.  Pulling the original stream from MediaMTX.
    2.  Decoding the video (FFmpeg).
    3.  Running the AI model (InsightFace).
    4.  Encoding the video (FFmpeg).
    5.  Pushing to the target RTMP server.

## Requirements

*   Linux (Ubuntu 20.04+ recommended)
*   Python 3.8+
*   NVIDIA GPU + CUDA Toolkit (for AI inference and hardware encoding)
*   FFmpeg (must support `h264_nvenc` if using GPU encoding)
*   MediaMTX

## Installation

1.  **Create a virtual environment and install dependencies**
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    ```
    *(Note: Ensure packages like `insightface`, `onnxruntime-gpu`, `fastapi`, `uvicorn`, `opencv-python` are installed)*

2.  **Download and Configure MediaMTX**
    Download MediaMTX and ensure `mediamtx.yml` is in the project root directory.
    ```bash
    # Example
    wget https://github.com/bluenviron/mediamtx/releases/download/v1.9.3/mediamtx_v1.9.3_linux_amd64.tar.gz
    tar -xvf mediamtx_v1.9.3_linux_amd64.tar.gz
    ```

3.  **Prepare AI Models**
    Ensure model files are located in the `.assets/models/` directory:
    *   `dynamic_batch_model.onnx` (InSwapper model)
    *   `buffalo_l` (InsightFace model pack, usually downloaded automatically)

## Starting the Service

You need to open two terminal windows simultaneously (or use systemd/supervisord for management):

**1. Start MediaMTX**
```bash
./mediamtx mediamtx.yml
```

**2. Start Backend API**
```bash
source .venv/bin/activate
python -m app.main
```

## Usage

### 1. Basic Streaming (Process and Return to Server)

Users stream via OBS or FFmpeg to:
`rtmp://<SERVER_IP>:1935/live/<STREAM_KEY>`

*   **Input**: `rtmp://<SERVER_IP>:1935/live/user1`
*   **Output**: The system automatically generates `rtmp://<SERVER_IP>:1935/live/user1_ai` for viewers.

### 2. Restream to External Platforms (e.g., YouTube)

Users can specify the destination URL (`target`) via the URL Query String when streaming.

**OBS Settings Example**:
*   **Server**: `rtmp://<SERVER_IP>:1935/live`
*   **Stream Key**: `user1?target=rtmp://a.rtmp.youtube.com/live2/YOUR_YOUTUBE_KEY`

**FFmpeg Test Example**:
```bash
ffmpeg -re -i input.mp4 -c:v libx264 -f flv "rtmp://localhost:1935/live/test?target=rtmp://a.rtmp.youtube.com/live2/KEY"
```

## Development & Debugging

*   **Check Logs**: The backend terminal will show detailed Worker startup, FPS information, and error messages.
*   **Test Webhook**: You can manually test the API using curl.
    ```bash
    curl -X POST -d "path=live/test" http://localhost:8000/on_publish
    ```
*   **No GPU Mode**: If testing in a non-GPU environment, modify `app/services/worker.py` to change `h264_nvenc` to `libx264` and adjust the ONNX Runtime Provider to `CPUExecutionProvider`.

## Notes

*   **Port Settings**: Default MediaMTX uses 1935 (RTMP) and 8000 (API Webhook). Ensure firewall ports are open.
*   **Multiprocessing**: Each stream starts an independent Process. Monitor GPU VRAM usage, as too many streams may cause OOM (Out Of Memory).

uvicorn app.main:app --reload