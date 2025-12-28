from fastapi import APIRouter, HTTPException, Query
from typing import Optional
from datetime import datetime

from app.schemas.session import (
    SessionCreate, SessionUpdate, UpdateSourceFace,
    SessionResponse, SessionListResponse, StreamStats,
    ApiResponse, StreamStatus
)
from app.services.session_service import session_manager
from app.services.process_manager import process_manager
from app.core import logger

router = APIRouter(prefix="/api/sessions", tags=["Sessions"])


@router.post("", response_model=ApiResponse)
async def create_session(request: SessionCreate):
    """
    創建或更新用戶推流會話配置
    
    用戶在開始推流前，先調用此 API 設置：
    - output_url: 目標平台的 RTMP 地址（如 Twitch, YouTube）
    - source_face_url: 換臉來源圖片
    - 其他視頻參數
    
    之後用戶使用 OBS 推流到:
    rtmp://{server}:1935/{api_key}/{api_secret}
    """
    try:
        session = session_manager.create_session(
            api_key=request.api_key,
            api_secret=request.api_secret,
            config=request.config
        )
        
        return ApiResponse(
            success=True,
            message="Session created/updated successfully",
            data={
                "session_id": session.session_id,
                "input_rtmp_url": f"rtmp://{{server}}:1935/{request.api_key}/{request.api_secret}",
                "output_url": request.config.output_url,
                "status": session.status.value
            }
        )
    except Exception as e:
        logger.error(f"[API] Failed to create session: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("", response_model=SessionListResponse)
async def list_sessions(api_key: Optional[str] = Query(None, description="Filter by API key")):
    """
    列出所有會話或指定 API key 的會話
    """
    sessions = session_manager.list_sessions(api_key)
    
    return SessionListResponse(
        sessions=[
            SessionResponse(
                session_id=s.session_id,
                api_key=s.api_key,
                status=s.status,
                input_rtmp_url=f"rtmp://{{server}}:1935/{s.api_key}/{s.api_secret}",
                output_url=s.config.output_url,
                source_face_url=s.config.source_face_url,
                created_at=s.created_at,
                updated_at=s.updated_at,
                config=s.config.model_dump()
            )
            for s in sessions
        ],
        total=len(sessions)
    )


@router.get("/{session_id}", response_model=SessionResponse)
async def get_session(session_id: str):
    """
    獲取指定會話的詳細信息
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    return SessionResponse(
        session_id=session.session_id,
        api_key=session.api_key,
        status=session.status,
        input_rtmp_url=f"rtmp://{{server}}:1935/{session.api_key}/{session.api_secret}",
        output_url=session.config.output_url,
        source_face_url=session.config.source_face_url,
        created_at=session.created_at,
        updated_at=session.updated_at,
        config=session.config.model_dump()
    )


@router.patch("/{session_id}", response_model=ApiResponse)
async def update_session(session_id: str, request: SessionUpdate):
    """
    更新會話配置
    
    可以更新 output_url, source_face_url 等設定
    注意：如果推流正在進行中，某些設定可能需要重啟 worker
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    update_data = request.model_dump(exclude_none=True)
    if not update_data:
        raise HTTPException(status_code=400, detail="No update data provided")
    
    # 更新會話配置
    session_manager.update_session(session_id, **update_data)
    
    return ApiResponse(
        success=True,
        message="Session updated successfully",
        data={"session_id": session_id, "updated_fields": list(update_data.keys())}
    )


@router.post("/{session_id}/source-face", response_model=ApiResponse)
async def update_source_face(session_id: str, request: UpdateSourceFace):
    """
    更新換臉來源圖片（支持熱更新）
    
    這個 API 會：
    1. 更新會話配置中的 source_face_url
    2. 如果有正在運行的 worker，發送消息讓其更新 source face
    
    用戶可以在直播過程中即時切換換臉來源
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # 更新配置
    session_manager.update_session(session_id, source_face_url=request.source_face_url)
    
    # 如果有正在運行的 worker，發送更新消息
    path = f"{session.api_key}/{session.api_secret}"
    if session.status == StreamStatus.ACTIVE:
        success = process_manager.send_message(path, {
            "type": "update_source_face",
            "url": request.source_face_url
        })
        if success:
            logger.info(f"[API] Sent source face update to worker: {session_id}")
        else:
            logger.warning(f"[API] Failed to send update to worker (might not be running): {session_id}")
    
    return ApiResponse(
        success=True,
        message="Source face updated successfully",
        data={
            "session_id": session_id,
            "source_face_url": request.source_face_url,
            "worker_notified": session.status == StreamStatus.ACTIVE
        }
    )


@router.delete("/{session_id}", response_model=ApiResponse)
async def delete_session(session_id: str):
    """
    刪除會話
    
    如果有正在運行的 worker，會先停止它
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    # 如果有運行中的 worker，先停止
    path = f"{session.api_key}/{session.api_secret}"
    process_manager.stop_process(path)
    
    # 刪除會話
    session_manager.delete_session(session_id)
    
    return ApiResponse(
        success=True,
        message="Session deleted successfully",
        data={"session_id": session_id}
    )


@router.get("/{session_id}/stats", response_model=StreamStats)
async def get_stream_stats(session_id: str):
    """
    獲取推流統計信息
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    stats = session.stats
    
    return StreamStats(
        session_id=session_id,
        status=session.status,
        uptime_seconds=stats.get("uptime_seconds"),
        frames_processed=stats.get("frames_processed"),
        current_fps=stats.get("current_fps"),
        input_rtmp_url=f"rtmp://{{server}}:1935/{session.api_key}/{session.api_secret}",
        output_url=session.config.output_url
    )


@router.post("/{session_id}/stop", response_model=ApiResponse)
async def stop_stream(session_id: str):
    """
    手動停止推流處理
    
    用於緊急停止或測試
    """
    session = session_manager.get_session(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    
    path = f"{session.api_key}/{session.api_secret}"
    success = process_manager.stop_process(path)
    
    if success:
        session_manager.update_session_status(session_id, StreamStatus.STOPPED)
        return ApiResponse(
            success=True,
            message="Stream stopped successfully",
            data={"session_id": session_id}
        )
    else:
        return ApiResponse(
            success=False,
            message="No active stream found for this session",
            data={"session_id": session_id}
        )
