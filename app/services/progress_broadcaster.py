"""In-process pub/sub for video job progress — feeds WebSocket subscribers."""
import asyncio
import threading
from collections import defaultdict
from datetime import datetime, UTC
from typing import Any, Dict, Optional, Set


class ProgressBroadcaster:
    """Per-process progress bus.

    - Stores the latest published state per job (thread-safe via lock), so late
      WS subscribers immediately receive the current snapshot.
    - Fans out updates to asyncio.Queue subscribers living in the event loop.
    - Can be published to from either the event loop (`publish`) or a worker
      thread (`publish_threadsafe`).
    """

    def __init__(self) -> None:
        self._state_lock = threading.Lock()
        self._latest: Dict[str, dict] = {}
        self._subscribers: Dict[str, Set[asyncio.Queue]] = defaultdict(set)

    def snapshot(self, job_id: str) -> Optional[dict]:
        with self._state_lock:
            return dict(self._latest[job_id]) if job_id in self._latest else None

    def _merge_state(self, job_id: str, fields: Dict[str, Any]) -> dict:
        with self._state_lock:
            state = self._latest.setdefault(job_id, {"job_id": job_id})
            state.update(fields)
            state["updated_at"] = datetime.now(UTC).isoformat()
            return dict(state)

    def drop(self, job_id: str) -> None:
        with self._state_lock:
            self._latest.pop(job_id, None)

    async def subscribe(self, job_id: str) -> asyncio.Queue:
        queue: asyncio.Queue = asyncio.Queue(maxsize=64)
        self._subscribers[job_id].add(queue)
        snap = self.snapshot(job_id)
        if snap is not None:
            queue.put_nowait(snap)
        return queue

    async def unsubscribe(self, job_id: str, queue: asyncio.Queue) -> None:
        subs = self._subscribers.get(job_id)
        if subs is None:
            return
        subs.discard(queue)
        if not subs:
            self._subscribers.pop(job_id, None)

    async def publish(self, job_id: str, **fields: Any) -> dict:
        payload = self._merge_state(job_id, fields)
        for queue in list(self._subscribers.get(job_id, ())):
            try:
                queue.put_nowait(payload)
            except asyncio.QueueFull:
                try:
                    queue.get_nowait()
                    queue.put_nowait(payload)
                except Exception:
                    pass
        return payload

    def publish_threadsafe(
        self, loop: asyncio.AbstractEventLoop, job_id: str, **fields: Any
    ) -> None:
        """Publish from a non-asyncio thread (e.g. ThreadPoolExecutor callback)."""
        future = asyncio.run_coroutine_threadsafe(self.publish(job_id, **fields), loop)
        future.add_done_callback(lambda f: f.exception())


progress_broadcaster = ProgressBroadcaster()
