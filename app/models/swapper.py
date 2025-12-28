"""
Real-time face swapping model using InsightFace
"""
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from app.core import logger


class RealTimeSwapper:
    """
    Real-time face swapper using InsightFace.
    
    Supports:
    - Face detection and landmark extraction
    - Face swapping using InSwapper model
    - Single face or all faces swapping
    """
    
    def __init__(
        self, 
        providers: list, 
        face_analysis_name: str, 
        inswapper_path: str, 
        det_size: tuple = (320, 320), 
        ctx_id: int = 0
    ):
        """
        Initialize the face swapper.
        
        Args:
            providers: ONNX runtime providers (e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider'])
            face_analysis_name: Name of face analysis model (e.g., 'buffalo_l')
            inswapper_path: Path to the InSwapper ONNX model
            det_size: Detection size for face analysis
            ctx_id: Context ID for GPU
        """
        # Face detector + landmarks + recognition (for source face)
        self.app = FaceAnalysis(
            name=face_analysis_name, 
            allowed_modules=["detection", "landmark", "recognition"],
            providers=providers
        )
        
        # Face detector + landmarks only (for target faces, faster)
        self.app2 = FaceAnalysis(
            name=face_analysis_name, 
            allowed_modules=["detection", "landmark"],
            providers=providers
        )
        
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)
        self.app2.prepare(ctx_id=ctx_id, det_size=det_size)
        
        # InSwapper model
        self.swapper = insightface.model_zoo.get_model(inswapper_path, providers=providers)

    def get_source_face(self, img: np.ndarray):
        """
        Extract face embedding from source image.
        
        Args:
            img: Source image in BGR format
            
        Returns:
            Face object with embedding
            
        Raises:
            RuntimeError: If no face detected
        """
        faces = self.app.get(img)
        if not faces:
            raise RuntimeError("No face detected in source image.")
        # Choose largest face
        faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
        return faces[0]

    def swap_into(self, frame_bgr: np.ndarray, src_face, swap_all: bool = False) -> np.ndarray:
        """
        Swap faces in the target frame.
        
        Args:
            frame_bgr: Target frame in BGR format
            src_face: Source face object from get_source_face()
            swap_all: If True, swap all faces; if False, swap only the largest face
            
        Returns:
            Frame with swapped faces
        """
        faces = self.app2.get(frame_bgr)
        if not faces:
            return frame_bgr
            
        if not swap_all:
            # Only swap the largest face
            faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
            faces = [faces[0]]
            
        out = frame_bgr
        for f in faces:
            try:
                out = self.swapper.get(out, f, src_face, paste_back=True)
            except Exception:
                # If swap fails (edge cases), skip that face
                continue
        return out
