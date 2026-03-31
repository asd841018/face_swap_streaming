import os
import cv2
import time
import threading
from app.core import logger


class FrameReader(threading.Thread):
    """
    Continuously reads frames from an RTMP stream, keeping only the latest frame.
    Supports automatic reconnection on stream interruption.
    """

    MAX_RECONNECT_ATTEMPTS = 5
    RECONNECT_BACKOFF_BASE = 1.0  # seconds

    def __init__(self, url):
        super().__init__(daemon=True)
        self.url = url
        self.cap = cv2.VideoCapture(url)
        self.running = False
        self.latest_frame = None
        self.lock = threading.Lock()
        self.connected = self.cap.isOpened()

    def _try_reconnect(self) -> bool:
        for attempt in range(1, self.MAX_RECONNECT_ATTEMPTS + 1):
            wait = self.RECONNECT_BACKOFF_BASE * attempt
            logger.info(f"[FrameReader] Reconnect attempt {attempt}/{self.MAX_RECONNECT_ATTEMPTS} in {wait:.1f}s")
            time.sleep(wait)
            try:
                self.cap.release()
            except Exception:
                pass
            self.cap = cv2.VideoCapture(self.url)
            if self.cap.isOpened():
                logger.info("[FrameReader] Reconnected successfully.")
                return True
        return False

    def run(self):
        self.running = True
        while self.running:
            if not self.cap.isOpened():
                if not self._try_reconnect():
                    logger.error("[FrameReader] All reconnect attempts failed. Stopping.")
                    break

            ret, frame = self.cap.read()
            if not ret:
                logger.info(f"[FrameReader] PID {os.getpid()} Lost frame, attempting reconnect...")
                if not self._try_reconnect():
                    logger.error("[FrameReader] Reconnect failed after frame loss. Stopping.")
                    break
                continue

            with self.lock:
                self.latest_frame = frame

        self.running = False
        try:
            self.cap.release()
        except Exception:
            pass

    def get_latest_frame(self):
        with self.lock:
            return self.latest_frame

    def stop(self):
        self.running = False