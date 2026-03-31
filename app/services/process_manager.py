import multiprocessing
import os
import signal
import json
import tempfile
import threading
from typing import Any, Dict
from app.core import logger
from app.config import settings

PID_FILE = "active_workers.json"


class ProcessManager:
    _instance = None
    _init_done = False

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if ProcessManager._init_done:
            return
        ProcessManager._init_done = True

        self.active_processes: Dict[str, multiprocessing.Process] = {}
        self.stop_events: Dict[str, Any] = {}
        self.queues: Dict[str, multiprocessing.Queue] = {}
        self._pid_file_lock = threading.Lock()

        if multiprocessing.current_process().name == "MainProcess":
            self.cleanup_stale_processes()

    @classmethod
    def get_instance(cls) -> "ProcessManager":
        return cls()

    def _load_pids(self) -> Dict[str, int]:
        if os.path.exists(PID_FILE):
            try:
                with open(PID_FILE, 'r') as f:
                    return json.load(f)
            except Exception as e:
                logger.error(f"[ProcessManager] Failed to load PID file: {e}")
        return {}

    def _save_pids(self, pids: Dict[str, int]):
        try:
            with self._pid_file_lock:
                with tempfile.NamedTemporaryFile("w", delete=False, dir=".", suffix=".tmp") as tmp:
                    json.dump(pids, tmp)
                    tmp_path = tmp.name
                os.replace(tmp_path, PID_FILE)
        except Exception as e:
            logger.error(f"[ProcessManager] Failed to save PID file: {e}")

    def cleanup_stale_processes(self):
        """Kill leftover processes from a PID file written by a previous run."""
        pids = self._load_pids()
        if not pids:
            return
        logger.info(f"[ProcessManager] Cleaning up {len(pids)} stale processes...")
        for path, pid in pids.items():
            try:
                os.kill(pid, signal.SIGTERM)
                logger.info(f"[ProcessManager] Killed stale process {pid} for {path}")
            except ProcessLookupError:
                logger.info(f"[ProcessManager] Stale process {pid} already dead.")
            except PermissionError:
                logger.error(f"[ProcessManager] Permission denied killing {pid}.")
            except Exception as e:
                logger.error(f"[ProcessManager] Error killing {pid}: {e}")
        self._save_pids({})

    def start_process(self, path: str, target, args=()) -> bool:
        if path in self.active_processes:
            if self.active_processes[path].is_alive():
                logger.info(f"[ProcessManager] Already running: {path}")
                return False
            self._remove_entry(path)

        live_count = sum(1 for p in self.active_processes.values() if p.is_alive())
        if live_count >= settings.MAX_WORKERS:
            logger.warning(f"[ProcessManager] Max workers reached ({live_count}/{settings.MAX_WORKERS}), rejecting {path}")
            return False

        stop_event = multiprocessing.Event()
        queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=target,
            args=(stop_event, queue) + args,
            daemon=True,
        )
        process.start()

        self.active_processes[path] = process
        self.stop_events[path] = stop_event
        self.queues[path] = queue

        pids = self._load_pids()
        pids[path] = process.pid
        self._save_pids(pids)

        logger.info(f"[ProcessManager] Started {path} (PID: {process.pid})")
        return True

    def stop_process(self, path: str) -> bool:
        if path not in self.active_processes:
            return False
        logger.info(f"[ProcessManager] Stopping {path}...")
        self.stop_events[path].set()

        process = self.active_processes[path]
        process.join(timeout=5)
        if process.is_alive():
            logger.warning(f"[ProcessManager] Terminating unresponsive process: {path}")
            process.terminate()
            process.join(timeout=1)
            if process.is_alive():
                process.kill()

        self._remove_entry(path)

        pids = self._load_pids()
        pids.pop(path, None)
        self._save_pids(pids)

        logger.info(f"[ProcessManager] Stopped: {path}")
        return True

    def _remove_entry(self, path: str):
        self.active_processes.pop(path, None)
        self.stop_events.pop(path, None)
        self.queues.pop(path, None)

    def stop_all(self):
        """Stop all active worker processes."""
        for path in list(self.active_processes.keys()):
            self.stop_process(path)

    def send_message(self, path: str, message: Any) -> bool:
        queue = self.queues.get(path)
        if queue is None:
            return False
        try:
            queue.put(message)
            return True
        except Exception as e:
            logger.error(f"[ProcessManager] Failed to send message to {path}: {e}")
            return False

process_manager = ProcessManager.get_instance()
