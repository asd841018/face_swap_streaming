# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Real-time RTMP stream processing service: receives live streams via MediaMTX, applies AI face swapping or beauty/vintage filters using InsightFace + InSwapper ONNX models, and pushes processed video to output destinations (YouTube, Twitch, or local MediaMTX). Also supports offline video face swap jobs via API.

## Running the Service

```bash
# Install dependencies (Poetry, requires Python 3.10+)
poetry install

# Start MediaMTX (terminal 1)
./mediamtx mediamtx.yml

# Start FastAPI backend (terminal 2) — runs on port 7980
python -m app.main
```

No test suite exists. The `tests/` directory contains standalone experimental scripts, not automated tests.

## Architecture

```
MediaMTX (RTMP :1935)
    │ webhooks: /on_publish, /on_publish_done
    ▼
FastAPI Backend (:7980)
    ├── Stream Monitor (polls MediaMTX every 2s, auto-starts workers)
    ├── Session Manager (persists to user_sessions.json)
    ├── Process Manager (singleton, manages worker lifecycle)
    └── Video Job Manager (offline video swap queue)
    │
    ▼
Worker Processes (one per stream, via multiprocessing)
    FrameReader thread → RealTimeSwapper (InsightFace/ONNX on GPU) → FFmpeg h264_nvenc → output RTMP
```

### Key design patterns

- **Multiprocessing per stream**: Each active stream gets its own OS process with its own GPU model instance. Inter-process communication uses `multiprocessing.Queue` for hot config updates (source face, filter type, KOL mode).
- **Singletons**: `ProcessManager`, `SessionManager`, `VideoJobManager` all use the singleton pattern (`__new__` override).
- **Persistence via JSON files**: `user_sessions.json` for sessions, `active_workers.json` for PIDs (stale process cleanup on startup).
- **Stream monitor**: Async background task (`monitor_streams()`) polls MediaMTX API, compares state deltas, and sends update messages to running workers when config changes.
- **Frame drop strategy**: `FrameReader` keeps only the latest frame (no buffering), worker reads non-blocking to maintain real-time output.

### Route modules

| Router | Prefix | Purpose |
|--------|--------|---------|
| `routes/webhooks.py` | `/` | MediaMTX callbacks: `on_publish`, `on_publish_done` |
| `routes/sessions.py` | `/api/sessions` | Session CRUD, source face hot-update, stats, stop |
| `routes/system.py` | `/api/system` | Health, status (CPU/GPU), workers list, cleanup |
| `routes/video.py` | `/api/video` | Offline video face swap job queue + status polling |

### Service layer

- **`worker.py`** — Entry point for worker processes: `run_stream_process(path, input_url, output_url, stop_event, queue, ...)`. Loads models, runs frame loop, processes queue messages.
- **`process_manager.py`** — Starts/stops worker processes, tracks PIDs, enforces `MAX_WORKERS`.
- **`session_service.py`** — CRUD for sessions, persists to `user_sessions.json`, generates session IDs from `api_key/api_secret`.
- **`stream_service.py`** — Bridges webhooks/monitor to process manager; resolves config from sessions or external API.
- **`monitor.py`** — `async monitor_streams()` task; polls MediaMTX, fetches config from external Faceswap API (HMAC auth), sends update messages to workers.
- **`video_job_service.py`** — Manages offline video swap jobs with semaphore for GPU concurrency control.

### AI models

- **InsightFace** (`buffalo_l`): Face detection, recognition, and landmarks. Auto-downloads to `~/.insightface/models/`.
- **InSwapper** (`.assets/models/inswapper_128_fp16.onnx`): ONNX model for face swapping.
- Both loaded per-worker process. Each instance uses ~1-2 GB GPU memory.

### Filters (`app/utils/`)

- `color_filtering.py` — Cold/warm beauty filters
- `old_film.py` — Vintage film effect
- `deform.py` — Face deformation (cheek/chin adjustment via grid warping)

## Configuration

All settings in `app/config.py` via Pydantic `BaseSettings`, loaded from `.env` file. Key settings:

- `SWAPPING_MODEL_PATH` — Path to InSwapper ONNX model
- `MEDIAMTX_API_URL` — MediaMTX API endpoint (default: `http://localhost:9997/v3/paths/list`)
- `MAX_WORKERS` — Max concurrent stream workers (default: 5)
- `MAX_CONCURRENT_VIDEO_SWAP_JOBS` — GPU semaphore for offline jobs (default: 1)
- `S3_UPLOAD_BUCKET`, `AWS_*` — S3 credentials for video job output
- `FACESWAP_API_BASE_URL` — External API for fetching stream config (HMAC authenticated)

## GPU/CPU Mode

The codebase assumes NVIDIA GPU with CUDA. To run on CPU, change in worker code:
- ONNX provider: `CUDAExecutionProvider` → `CPUExecutionProvider`
- FFmpeg encoder: `h264_nvenc` → `libx264`
