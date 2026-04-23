import asyncio
import aiohttp
from typing import Any, Dict
from app.services.stream_service import stream_service
from app.services.session_service import session_manager
from app.services.process_manager import process_manager
from app.core import logger
from app.utils import FaceswapApiClient
from app.config import settings

# Fields that can be hot-updated on a running worker via queue messages.
_UPDATABLE_FIELDS = [
    ("source_face_url", "update_source_face", "url"),
    ("use_image_filter", "update_use_image_filter", "use_image_filter"),
    ("filter_type", "update_filter_type", "filter_type"),
]


def _parse_external_payload(api_response: Dict[str, Any]) -> Dict[str, Any]:
    data = api_response.get("data", {}) if isinstance(api_response, dict) else {}
    return {
        "source_face_url": data.get("face_image_url"),
        "use_image_filter": bool(data.get("use_image_filter", False)),
        "filter_type": data.get("filter_type"),
        "is_kol_mode": bool(data.get("is_kol_mode", False)),
        "kol_source_url": data.get("kol_source_url"),
    }


def _config_from_session(session) -> Dict[str, Any]:
    cfg = session.config
    return {
        "source_face_url": cfg.source_face_url,
        "use_image_filter": cfg.use_image_filter,
        "is_kol_mode": cfg.is_kol_mode,
        "kol_source_url": cfg.kol_source_url,
        "filter_type": cfg.filter_type.value if cfg.filter_type else None,
    }


_DEFAULT_CONFIG: Dict[str, Any] = {
    "source_face_url": None,
    "use_image_filter": False,
    "filter_type": None,
    "is_kol_mode": False,
    "kol_source_url": None,
}


async def _fetch_from_base(
    base_url: str,
    api_key: str,
    api_secret: str,
    order_number: str,
    http: aiohttp.ClientSession,
) -> Dict[str, Any]:
    client = FaceswapApiClient(base_url=base_url, api_key=api_key, api_secret=api_secret)
    resp = await client.get_face_image_async(
        session=http,
        order_number=order_number,
        timeout_seconds=settings.FACESWAP_API_TIMEOUT_SECONDS,
    )
    return _parse_external_payload(resp)


async def _fetch_external_config(path: str, http: aiohttp.ClientSession) -> Dict[str, Any]:
    parts = path.split("/")
    if len(parts) < 2:
        return dict(_DEFAULT_CONFIG)
    api_key, api_secret = parts[0], parts[1]
    order_number = parts[2] if len(parts) > 2 else ""

    base_urls = [
        ("dev", settings.FACESWAP_DEV_BASE_URL),
        ("prd", settings.FACESWAP_PRD_BASE_URL),
    ]
    candidates = [(label, url) for label, url in base_urls if url]
    if not candidates:
        return dict(_DEFAULT_CONFIG)

    results = await asyncio.gather(
        *(_fetch_from_base(url, api_key, api_secret, order_number, http) for _, url in candidates),
        return_exceptions=True,
    )
    for (label, url), result in zip(candidates, results):
        if isinstance(result, dict):
            logger.info(f"[Monitor] Fetched config for {path} from {label} ({url})")
            return result
    for (label, _), result in zip(candidates, results):
        logger.error(f"[Monitor] Fetch config failed for {path} from {label}: {result}")
    return dict(_DEFAULT_CONFIG)


def _send_updates(path: str, old: Dict, new: Dict):
    """Compare old/new state and send queue messages for changed fields."""
    for field, msg_type, payload_key in _UPDATABLE_FIELDS:
        if new.get(field) != old.get(field):
            logger.info(f"[Monitor] {field} changed for {path}")
            process_manager.send_message(path, {"type": msg_type, payload_key: new[field]})

    if new.get("is_kol_mode") != old.get("is_kol_mode"):
        logger.info(f"[Monitor] KOL mode changed for {path}")
        process_manager.send_message(path, {
            "type": "update_kol_mode",
            "is_kol_mode": new["is_kol_mode"],
            "kol_source_url": new.get("kol_source_url"),
        })
    elif new.get("kol_source_url") != old.get("kol_source_url"):
        logger.info(f"[Monitor] KOL source URL changed for {path}")
        process_manager.send_message(path, {
            "type": "update_kol_source_url",
            "kol_source_url": new.get("kol_source_url"),
        })


async def monitor_streams():
    """Poll MediaMTX for active streams, start/update/stop workers accordingly."""
    logger.info("[Monitor] Starting stream monitor...")
    stream_states: Dict[str, Dict[str, Any]] = {}

    async with aiohttp.ClientSession() as http:
        while True:
            try:
                async with http.get(settings.MEDIAMTX_API_URL) as resp:
                    if resp.status != 200:
                        logger.warning(f"[Monitor] MediaMTX API returned {resp.status}")
                        await asyncio.sleep(settings.MONITOR_POLL_INTERVAL_SECONDS)
                        continue

                    data = await resp.json()
                    items = data.get("items") or []
                    ready_paths = {it["name"] for it in items if it.get("ready") and it.get("name")}
                    # Filter out output streams: first segment ends with "_ai"
                    # e.g. "apikey_ai" or "apikey_ai/001"
                    source_paths = [p for p in ready_paths if not p.split("/")[0].endswith("_ai")]

                    # Build config for each source path
                    configs: Dict[str, Dict[str, Any]] = {}
                    fetch_paths, fetch_tasks = [], []
                    for p in source_paths:
                        session = session_manager.get_session_for_stream(p)
                        if session:
                            configs[p] = _config_from_session(session)
                        else:
                            fetch_paths.append(p)
                            fetch_tasks.append(_fetch_external_config(p, http))
                    if fetch_tasks:
                        results = await asyncio.gather(*fetch_tasks)
                        for p, cfg in zip(fetch_paths, results):
                            configs[p] = cfg
                    # Start new / update existing workers
                    for p in source_paths:
                        cfg = configs.get(p, dict(_DEFAULT_CONFIG))
                        is_running = p in process_manager.active_processes

                        if not is_running:
                            if stream_service.start_worker(path=p, startup_overrides=cfg):
                                stream_states[p] = cfg.copy()
                                logger.info(f"[Monitor] Started worker for {p}")
                        else:
                            prev = stream_states.get(p)
                            if prev is None:
                                stream_states[p] = cfg.copy()
                                prev = stream_states[p]
                            _send_updates(p, prev, cfg)
                            stream_states[p] = cfg.copy()

                    # Stop workers for disappeared streams
                    for wp in list(process_manager.active_processes):
                        if wp not in ready_paths:
                            logger.info(f"[Monitor] Stream {wp} ended, stopping worker.")
                            stream_service.stop_worker(wp)
                            stream_states.pop(wp, None)

            except asyncio.CancelledError:
                logger.info("[Monitor] Task cancelled.")
                raise
            except aiohttp.ClientConnectorError:
                logger.error("[Monitor] Cannot connect to MediaMTX API.")
            except Exception as e:
                logger.error(f"[Monitor] Error: {e}")

            await asyncio.sleep(settings.MONITOR_POLL_INTERVAL_SECONDS)
