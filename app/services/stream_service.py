from app.services.process_manager import process_manager
from app.services.worker import run_stream_process
from app.services.session_service import session_manager
from app.schemas.session import StreamStatus
from app.core import logger
from app.config import settings
from urllib.parse import parse_qs
from typing import Any, Dict, Optional


class StreamService:
    def __init__(self):
        self.mediamtx_host = "127.0.0.1"
        self.mediamtx_port = "1935"

    @staticmethod
    def _to_bool(value: Optional[str]) -> bool:
        if value is None:
            return False
        return str(value).strip().lower() in {"1", "true", "yes", "on"}

    def _default_output(self, path: str) -> str:
        api_key = path.split("/")[0]
        return f"rtmp://{self.mediamtx_host}:{self.mediamtx_port}/{api_key}_ai"

    def _parse_query(self, query: Optional[str]) -> Dict[str, Any]:
        if not query:
            return {}
        qs = parse_qs(query)

        def first(keys):
            for k in keys:
                vals = qs.get(k)
                if vals:
                    return vals[0]
            return None

        fr_raw = first(["frame_rate"])
        try:
            fr = int(fr_raw) if fr_raw else None
        except ValueError:
            fr = None

        return {
            "output_url": first(["target", "push_url"]),
            "source_face_url": first(["source_face", "source_face_url"]),
            "filter_type": first(["filter_type"]),
            "use_image_filter": self._to_bool(first(["use_image_filter"])),
            "video_bitrate": first(["video_bitrate"]),
            "video_resolution": first(["video_resolution"]),
            "frame_rate": fr,
            "swap_all": self._to_bool(first(["swap_all"])),
        }

    def start_worker(
        self,
        path: str,
        query: Optional[str] = None,
        startup_overrides: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if path.endswith("_ai"):
            return False

        ov = startup_overrides or {}
        qo = self._parse_query(query)
        session = session_manager.get_session_for_stream(path)

        # Resolve config — session > override > query > default
        if session:
            sc = session.config
            output_rtmp = sc.output_url
            source_face_url = sc.source_face_url or ov.get("source_face_url") or qo.get("source_face_url")
            use_image_filter = sc.use_image_filter
            filter_type = sc.filter_type.value if sc.filter_type else ov.get("filter_type") or qo.get("filter_type")
            is_kol_mode = sc.is_kol_mode if hasattr(sc, "is_kol_mode") else False
            kol_source_url = sc.kol_source_url if hasattr(sc, "kol_source_url") else None
            video_config = {
                "bitrate": sc.video_bitrate or settings.DEFAULT_VIDEO_BITRATE,
                "resolution": sc.video_resolution or settings.DEFAULT_VIDEO_RESOLUTION,
                "frame_rate": sc.frame_rate or settings.DEFAULT_FRAME_RATE,
                "swap_all": sc.swap_all,
            }
            logger.info(f"[Service] Using session: {session.session_id}")
        else:
            output_rtmp = ov.get("output_url") or qo.get("output_url")
            source_face_url = ov.get("source_face_url") or qo.get("source_face_url")
            use_image_filter = ov.get("use_image_filter", qo.get("use_image_filter", False))
            filter_type = ov.get("filter_type") or qo.get("filter_type")
            is_kol_mode = ov.get("is_kol_mode", qo.get("is_kol_mode", False))
            kol_source_url = ov.get("kol_source_url", qo.get("kol_source_url"))
            video_config = {
                "bitrate": qo.get("video_bitrate") or settings.DEFAULT_VIDEO_BITRATE,
                "resolution": qo.get("video_resolution") or settings.DEFAULT_VIDEO_RESOLUTION,
                "frame_rate": qo.get("frame_rate") or settings.DEFAULT_FRAME_RATE,
                "swap_all": qo.get("swap_all", False),
            }

        if not output_rtmp:
            output_rtmp = self._default_output(path)

        input_rtmp = f"rtmp://{self.mediamtx_host}:{self.mediamtx_port}/{path}"
        logger.info(f"[Service] Starting worker: {input_rtmp} -> {output_rtmp}")

        try:
            ok = process_manager.start_process(
                path,
                run_stream_process,
                args=(input_rtmp, output_rtmp, source_face_url, use_image_filter, filter_type, is_kol_mode, kol_source_url, video_config),
            )
            if ok and session:
                session_manager.update_session_status(session.session_id, StreamStatus.ACTIVE)
            return ok
        except Exception as e:
            logger.error(f"[Service] Failed to start worker: {e}")
            if session:
                session_manager.update_session_status(session.session_id, StreamStatus.ERROR)
            return False

    def stop_worker(self, path: str) -> bool:
        session = session_manager.get_session_for_stream(path)
        if session:
            session_manager.update_session_status(session.session_id, StreamStatus.STOPPED)
        return process_manager.stop_process(path)


stream_service = StreamService()
