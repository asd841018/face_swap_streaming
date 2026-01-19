from app.services.process_manager import process_manager
from app.services.worker import run_stream_process
from app.services.session_service import session_manager
from app.schemas.session import StreamStatus
from app.core import logger

class StreamService:
    def __init__(self):
        self.mediamtx_host = "127.0.0.1"
        self.mediamtx_port = "1935"

    def start_worker(self, path: str, query: str = None) -> bool:
        """
        Start a worker process for the given stream path.
        
        優先使用用戶預先配置的會話設定，
        如果沒有預先配置，則使用 query 參數或默認值
        """
        # Avoid infinite loop: ignore streams already processed by AI
        if path.endswith("_ai"):
            logger.info(f"[Service] Ignoring AI stream: {path}")
            return False

        # 1. 嘗試從會話管理器獲取預先配置的設定
        session = session_manager.get_session_for_stream(path)
        
        output_rtmp = None
        source_face_url = None
        video_config = {}
        
        if session:
            # 使用預先配置的設定
            output_rtmp = session.config.output_url
            source_face_url = session.config.source_face_url
            video_config = {
                "bitrate": session.config.video_bitrate,
                "resolution": session.config.video_resolution,
                "frame_rate": session.config.frame_rate,
                "swap_all": session.config.swap_all
            }
            logger.info(f"[Service] Using pre-configured session: {session.session_id}")
            
            # 更新會話狀態為 ACTIVE
            session_manager.update_session_status(session.session_id, StreamStatus.ACTIVE)
        else:
            # 2. 沒有預先配置，嘗試從 query 參數獲取
            if query:
                from urllib.parse import parse_qs
                qs = parse_qs(query)
                if 'target' in qs:
                    output_rtmp = qs['target'][0]
                elif 'push_url' in qs:
                    output_rtmp = qs['push_url'][0]
                if 'source_face' in qs:
                    source_face_url = qs['source_face'][0]
            
            # 3. 如果還是沒有 output_rtmp，使用默認值（本地 AI 流）
            if not output_rtmp:
                API_KEY = path.split('/')[0]
                output_rtmp = f"rtmp://{self.mediamtx_host}:{self.mediamtx_port}/{API_KEY}_ai"
                logger.warning(f"[Service] No output URL configured, using default: {output_rtmp}")

        input_rtmp = f"rtmp://{self.mediamtx_host}:{self.mediamtx_port}/{path}"

        logger.info(f"[Service] Requesting Worker Process: {input_rtmp} -> {output_rtmp}")

        try:
            # Use ProcessManager to start a new process
            success = process_manager.start_process(
                path, 
                run_stream_process, 
                args=(input_rtmp, output_rtmp, source_face_url, video_config)
            )
            return success
            
        except Exception as e:
            logger.error(f"[Service] Failed to start worker: {e}")
            if session:
                session_manager.update_session_status(session.session_id, StreamStatus.ERROR)
            return False

    def stop_worker(self, path: str) -> bool:
        """
        Stop the worker process for the given stream path.
        """
        # 更新會話狀態
        session = session_manager.get_session_for_stream(path)
        if session:
            session_manager.update_session_status(session.session_id, StreamStatus.STOPPED)
        
        return process_manager.stop_process(path)

    def start_worker_legacy(self, path: str, 
                            source_face_url: str = None, 
                            ref_image_url: str = None,
                            use_image_filter: bool = False) -> bool:
        """
        舊版本的 start_worker，用於沒有預先配置會話的情況。
        使用默認的 output_rtmp（本地 AI 流）
        """
        if path.endswith("_ai"):
            logger.info(f"[Service] Ignoring AI stream: {path}")
            return False
        
        API_KEY = path.split('/')[0]
        output_rtmp = f"rtmp://{self.mediamtx_host}:{self.mediamtx_port}/{API_KEY}_ai"
        input_rtmp = f"rtmp://{self.mediamtx_host}:{self.mediamtx_port}/{path}"

        logger.info(f"[Service] (Legacy) Requesting Worker Process: {input_rtmp} -> {output_rtmp}")

        try:
            success = process_manager.start_process(
                path, 
                run_stream_process, 
                args=(input_rtmp, output_rtmp, source_face_url,  ref_image_url, use_image_filter, {})
            )
            return success
        except Exception as e:
            logger.error(f"[Service] Failed to start worker: {e}")
            return False

# Global instance
stream_service = StreamService()
