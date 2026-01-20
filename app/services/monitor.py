import asyncio
import aiohttp
from app.services.stream_service import stream_service
from app.services.session_service import session_manager
from app.schemas.session import StreamStatus
from app.core import logger
from app.utils import FaceswapApiClient

MEDIAMTX_API_URL = "http://localhost:9997/v3/paths/list"
BASE_URL = "https://api.aimate.am"

async def monitor_streams():
    """
    Periodically poll MediaMTX API to check for active streams.
    
    流程:
    1. 每 2 秒輪詢 MediaMTX API 獲取活躍的 RTMP 流
    2. 對於新流:
       - 優先使用本地會話配置 (用戶通過 API 預先設定的)
       - 如果沒有本地配置，從外部 API 獲取 source_face_url
    3. 對於已存在的流，檢查是否需要更新 source_face
    4. 對於已結束的流，停止 worker
    """
    logger.info("[Monitor] Starting stream monitor...")
    stream_states = {} # {path: source_face_url}

    async with aiohttp.ClientSession() as session:
        while True:
            try:
                async with session.get(MEDIAMTX_API_URL) as response:
                    if response.status == 200:
                        data = await response.json()
                        # data format: {'items': [{'name': 'live/user1', ...}, ...]}
                        current_paths = set()
                        
                        # A list of all active stream paths, includes the output "_ai" streams
                        items = data.get("items", [])
                        for item in items:
                            path_name = item.get("name")
                            # Check if the path has active publishers (source is ready)
                            if item.get("ready", False): 
                                current_paths.add(path_name)

                        # 1. Start workers for new streams or update existing ones
                        for path in current_paths:
                            if path.endswith("_ai"): # Ignore our own output streams
                                continue
                            
                            # Parse user_id and user_secret_key from path
                            # Expected format: {user_id}/{user_secret_key}
                            parts = path.split('/')
                            API_KEY = parts[0] if len(parts) >= 1 else None
                            API_SECRET = parts[1] if len(parts) >= 2 else None
                            
                            # 檢查是否有本地會話配置
                            # Fix: 這一段永遠是None
                            local_session = session_manager.get_session_for_stream(path)
                            
                            source_face_url = None
                            ref_image_url = None
                            use_image_filter = False
                            use_local_config = False
                            
                            if local_session:
                                # 使用本地預設的配置
                                source_face_url = local_session.config.source_face_url
                                use_local_config = True
                                logger.debug(f"[Monitor] Using local session config for {path}")
                            else:
                                # 沒有本地配置，從外部 API 獲取
                                try:
                                    client = FaceswapApiClient(base_url=BASE_URL, 
                                                               api_key=API_KEY, 
                                                               api_secret=API_SECRET)
                                    api_response = client.get_face_image()
                                    data = api_response.get("data", {})
                                    if data:
                                        source_face_url = data.get('face_image_url', None)
                                        ref_image_url = data.get('ref_image_url', None)
                                        use_image_filter = data.get('use_image_filter', False)
                                        logger.debug(f"[Monitor] Fetched face image for {path} from external API")
                                except Exception as e:
                                    logger.error(f"[Monitor] Failed to fetch face image for {path}: {e}")

                            # Check if worker is already running
                            from app.services.process_manager import process_manager
                            is_running = path in process_manager.active_processes

                            if not is_running:
                                # Start new worker
                                if use_local_config:
                                    print("-------------------- Using local config ------------------")
                                    # 使用本地會話配置啟動（包含 output_url 等）
                                    success = stream_service.start_worker(path=path, query=None)
                                else:
                                    # 沒有本地配置，使用舊的方式
                                    success = stream_service.start_worker_legacy(
                                        path=path, 
                                        source_face_url=source_face_url,
                                        ref_image_url=ref_image_url,
                                        use_image_filter=use_image_filter
                                    )
                                
                                if success:
                                    stream_states[path] = {"source_face": source_face_url}
                                    stream_states[path] = {"ref_image": ref_image_url}
                                    stream_states[path] = {"use_image_filter": use_image_filter}
                                    logger.info(f"[Monitor] Started worker for new stream: {path}")
                            else:
                                # Worker is running, check for update
                                # 只在非本地配置模式下檢查外部 API 的更新
                                # 本地配置的更新通過 /api/sessions/{id}/source-face API 處理
                                if not use_local_config:
                                    last_data = stream_states.get(path)
                                    if source_face_url != last_data.get("source_face"):
                                        # logger.info(f"[Monitor] Detected source face change for {path}. Updating worker.")
                                        process_manager.send_message(path, {
                                            'type': 'update_source_face',
                                            'url': source_face_url
                                        })
                                        stream_states[path] = {"source_face": source_face_url}
                                    if ref_image_url != last_data.get("ref_image"):
                                        # logger.info(f"[Monitor] Detected ref image change for {path}. Updating worker.")
                                        process_manager.send_message(path, {
                                            'type': 'update_ref_image',
                                            'url': ref_image_url
                                        })
                                        stream_states[path] = {"ref_image": ref_image_url}
                                    if use_image_filter != last_data.get("use_image_filter"):
                                        # logger.info(f"[Monitor] Detected image filter change for {path}. Updating worker.")
                                        process_manager.send_message(path, {
                                            'type': 'update_use_image_filter',
                                            'use_image_filter': use_image_filter
                                        })
                                        stream_states[path] = {"use_image_filter": use_image_filter}

                        # 2. Stop workers for ended streams
                        # Get all active worker paths from stream_service/process_manager
                        # We need a way to know which workers are running. 
                        # Let's assume stream_service tracks them or we can ask process_manager.
                        from app.services.process_manager import process_manager
                        active_workers = list(process_manager.active_processes.keys())
                        
                        for worker_path in active_workers:
                            if worker_path not in current_paths:
                                logger.info(f"[Monitor] Stream {worker_path} ended. Stopping worker.")
                                stream_service.stop_worker(worker_path)
                                if worker_path in stream_states:
                                    del stream_states[worker_path]

                    else:
                        logger.warning(f"[Monitor] MediaMTX API returned {response.status}")
                        try:
                            error_text = await response.text()
                            logger.warning(f"[Monitor] API Response: {error_text}")
                        except:
                            pass

            except aiohttp.ClientConnectorError:
                logger.error("[Monitor] Cannot connect to MediaMTX API. Is it running?")
            except Exception as e:
                logger.error(f"[Monitor] Error: {e}")

            await asyncio.sleep(2) # Poll every 2 seconds
