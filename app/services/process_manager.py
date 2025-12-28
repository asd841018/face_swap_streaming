import multiprocessing
import time
import os
import signal
import json
from typing import Dict, Any
from app.core import logger

PID_FILE = "active_workers.json"


class ProcessManager:
    _instance = None

    def __init__(self):
        self.active_processes: Dict[str, multiprocessing.Process] = {}
        self.stop_events: Dict[str, Any] = {}
        self.queues: Dict[str, multiprocessing.Queue] = {}
        
        # Only cleanup in main process, not in child processes spawned by multiprocessing
        if multiprocessing.current_process().name == 'MainProcess':
            self.cleanup_stale_processes()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = ProcessManager()
        return cls._instance

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
            with open(PID_FILE, 'w') as f:
                json.dump(pids, f)
        except Exception as e:
            logger.error(f"[ProcessManager] Failed to save PID file: {e}")

    def cleanup_stale_processes(self):
        """Kill any processes listed in the PID file that might be left over from a previous run."""
        pids = self._load_pids()
        if not pids:
            return

        logger.info(f"[ProcessManager] Cleaning up {len(pids)} stale processes...")
        for path, pid in pids.items():
            try:
                # Check if process exists and kill it
                os.kill(pid, signal.SIGTERM)
                logger.info(f"[ProcessManager] Killed stale process {pid} for {path}")
            except ProcessLookupError:
                logger.info(f"[ProcessManager] Stale process {pid} for {path} not found (already dead).")
            except PermissionError:
                logger.error(f"[ProcessManager] Permission denied killing process {pid}.")
            except Exception as e:
                logger.error(f"[ProcessManager] Error killing process {pid}: {e}")
        
        # Clear the file
        self._save_pids({})

    def start_process(self, path: str, target, args=()):
        if path in self.active_processes:
            if self.active_processes[path].is_alive():
                logger.info(f"[ProcessManager] Process already running for: {path}")
                return False
            else:
                del self.active_processes[path]
                del self.stop_events[path]
                if path in self.queues:
                    del self.queues[path]

        stop_event = multiprocessing.Event()
        queue = multiprocessing.Queue()
        # Pass stop_event and queue as the first two arguments to the target function
        process_args = (stop_event, queue) + args
        
        process = multiprocessing.Process(target=target, args=process_args, daemon=True)
        process.start()
        
        self.active_processes[path] = process
        self.stop_events[path] = stop_event
        self.queues[path] = queue
        
        # Update PID file
        pids = self._load_pids()
        pids[path] = process.pid
        self._save_pids(pids)

        logger.info(f"[ProcessManager] Started process for {path} (PID: {process.pid})")
        return True

    def stop_process(self, path: str):
        if path in self.active_processes:
            logger.info(f"[ProcessManager] Stopping process for {path}...")
            self.stop_events[path].set()
            
            process = self.active_processes[path]
            process.join(timeout=5)
            
            if process.is_alive():
                logger.warning(f"[ProcessManager] Process unresponsive, terminating: {path}")
                process.terminate()
                process.join(timeout=1)
                if process.is_alive():
                    process.kill()
            
            del self.active_processes[path]
            del self.stop_events[path]
            if path in self.queues:
                del self.queues[path]
            
            # Update PID file
            pids = self._load_pids()
            if path in pids:
                del pids[path]
                self._save_pids(pids)

            logger.info(f"[ProcessManager] Process stopped: {path}")
            return True
        return False

    def send_message(self, path: str, message: Any):
        if path in self.queues:
            try:
                self.queues[path].put(message)
                return True
            except Exception as e:
                logger.error(f"[ProcessManager] Failed to send message to {path}: {e}")
        return False

process_manager = ProcessManager.get_instance()
