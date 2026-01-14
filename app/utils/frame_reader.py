import os
import cv2
import threading
from app.core import logger

class FrameReader(threading.Thread):
    """
    Continuously reads frames from the capture object to ensure we always have the latest frame.
    """
    def __init__(self, url):
        super().__init__()
        self.url = url
        self.cap = cv2.VideoCapture(url)
        self.running = False
        self.latest_frame = None
        self.lock = threading.Lock()
        self.daemon = True
        self.connected = self.cap.isOpened()

    def run(self):
        self.running = True
        while self.running and self.cap.isOpened():
            ret, frame = self.cap.read()
            if not ret:
                self.running = False
                logger.info(f"[ProcessWorker] PID {os.getpid()} Cannot read frame from stream.")
                break
            with self.lock:
                self.latest_frame = frame
        self.cap.release()

    def get_latest_frame(self):
        with self.lock:
            return self.latest_frame

    def stop(self):
        self.running = False