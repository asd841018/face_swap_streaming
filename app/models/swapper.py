"""Real-time face swapping model using InsightFace."""
import numpy as np
import insightface
from insightface.app import FaceAnalysis
from app.core import logger
from app.config import settings
from app.schemas.session import FilterType
from app.utils.deform import reshape_faces
from app.utils.color_filtering import beauty_pipeline
from app.utils.old_film import vintage_filter


class RealTimeSwapper:
    """Real-time face swapper: detection, swap, and filter pipelines."""

    def __init__(self, providers: list, face_analysis_name: str, inswapper_path: str,
                 det_size: tuple = (640, 640), ctx_id: int = 0):
        self.app = FaceAnalysis(name=face_analysis_name,
                                allowed_modules=["detection", "recognition"],
                                providers=providers)
        self.app2 = FaceAnalysis(name=face_analysis_name,
                                 allowed_modules=["detection", "landmark_2d_106"],
                                 providers=providers)
        self.app.prepare(ctx_id=ctx_id, det_size=det_size)
        self.app2.prepare(ctx_id=ctx_id, det_size=det_size)
        self.swapper = insightface.model_zoo.get_model(inswapper_path, providers=providers)

    def get_source_face(self, img: np.ndarray):
        """Extract the largest face embedding from *img*."""
        faces = self.app.get(img)
        if not faces:
            raise RuntimeError("No face detected in source image.")
        faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
        return faces[0]

    def swap_into(self, frame_bgr: np.ndarray, src_face, swap_all: bool = False) -> np.ndarray:
        """Swap faces in *frame_bgr* with *src_face*."""
        faces = self.app2.get(frame_bgr)
        if not faces:
            return frame_bgr
        if not swap_all:
            faces.sort(key=lambda f: (f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1]), reverse=True)
            faces = [faces[0]]
        out = frame_bgr
        for f in faces:
            try:
                out = self.swapper.get(out, f, src_face, paste_back=True)
            except Exception:
                continue
        return out

    # --- Filter pipeline ---

    _REF_URL_MAP = None

    @classmethod
    def _ref_url_map(cls):
        if cls._REF_URL_MAP is None:
            cls._REF_URL_MAP = {
                settings.FILTER_REF_URL_COLD: FilterType.COLD_BEAUTY.value,
                settings.FILTER_REF_URL_WARM: FilterType.WARM_BEAUTY.value,
                settings.FILTER_REF_URL_VINTAGE: FilterType.VINTAGE.value,
            }
        return cls._REF_URL_MAP

    def _resolve_filter(self, filter_type: str | None, ref_image_url: str | None) -> str | None:
        valid = {ft.value for ft in FilterType}
        if filter_type in valid:
            return filter_type
        if ref_image_url:
            return self._ref_url_map().get(ref_image_url)
        return None

    def deform_face(self, frame_bgr: np.ndarray, ref_image_url: str | None = None,
                    filter_type: str | None = None) -> np.ndarray:
        """Apply face deform + beauty/vintage filter."""
        faces = self.app2.get(frame_bgr)
        if not faces:
            return frame_bgr

        ft = self._resolve_filter(filter_type, ref_image_url)
        if not ft:
            return frame_bgr

        result = reshape_faces(
            frame_bgr, faces,
            cheek_strength=settings.CHEEK_STRENGTH,
            chin_strength=settings.CHIN_STRENGTH,
            grid_resolution=settings.GRID_RESOLUTION,
        )
        if ft == FilterType.COLD_BEAUTY.value:
            return beauty_pipeline(frame=result, strength=0.45, brighten=5, smooth=0.35, mode="cold")
        if ft == FilterType.WARM_BEAUTY.value:
            return beauty_pipeline(frame=result, strength=0.45, brighten=5, smooth=0.35, mode="warm")
        if ft == FilterType.VINTAGE.value:
            return vintage_filter(result)
        return frame_bgr
