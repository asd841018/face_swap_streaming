from fastapi import APIRouter
from typing import Dict, Any
from datetime import datetime
import os

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

from app.services.process_manager import process_manager
from app.services.session_service import session_manager
from app.schemas.session import StreamStatus
from app.core import logger

router = APIRouter(prefix="/api/system", tags=["System"])


@router.get("/health")
async def health_check():
    """
    系統健康檢查
    """
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat()
    }


@router.get("/status")
async def system_status():
    """
    獲取系統整體狀態
    
    包括：
    - 活躍的 worker 數量
    - 會話統計
    - 系統資源使用
    """
    # 統計會話
    all_sessions = session_manager.list_sessions()
    active_sessions = [s for s in all_sessions if s.status == StreamStatus.ACTIVE]
    pending_sessions = [s for s in all_sessions if s.status == StreamStatus.PENDING]
    
    # 活躍的 worker
    active_workers = len(process_manager.active_processes)
    
    # 系統資源
    if HAS_PSUTIL:
        cpu_percent = psutil.cpu_percent()
        memory = psutil.virtual_memory()
        system_info = {
            "cpu_percent": cpu_percent,
            "memory_percent": memory.percent,
            "memory_used_gb": round(memory.used / (1024**3), 2),
            "memory_total_gb": round(memory.total / (1024**3), 2)
        }
    else:
        system_info = {"note": "psutil not installed"}
    
    # GPU 資訊（如果有 nvidia-smi）
    gpu_info = None
    try:
        import subprocess
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total', '--format=csv,noheader,nounits'],
            capture_output=True, text=True, timeout=5
        )
        if result.returncode == 0:
            parts = result.stdout.strip().split(', ')
            if len(parts) >= 3:
                gpu_info = {
                    "utilization_percent": int(parts[0]),
                    "memory_used_mb": int(parts[1]),
                    "memory_total_mb": int(parts[2])
                }
    except Exception:
        pass
    
    return {
        "timestamp": datetime.now().isoformat(),
        "sessions": {
            "total": len(all_sessions),
            "active": len(active_sessions),
            "pending": len(pending_sessions)
        },
        "workers": {
            "active": active_workers,
            "pids": list(process_manager.active_processes.keys())
        },
        "system": system_info,
        "gpu": gpu_info
    }


@router.get("/workers")
async def list_workers():
    """
    列出所有活躍的 worker 進程
    """
    workers = []
    for path, process in process_manager.active_processes.items():
        workers.append({
            "path": path,
            "pid": process.pid,
            "is_alive": process.is_alive()
        })
    
    return {
        "workers": workers,
        "total": len(workers)
    }


@router.post("/cleanup")
async def cleanup_stale_processes():
    """
    清理殭屍進程
    """
    process_manager.cleanup_stale_processes()
    return {
        "success": True,
        "message": "Cleanup completed"
    }
