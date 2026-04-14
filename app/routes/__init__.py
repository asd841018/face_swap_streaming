from app.routes.webhooks import router as webhooks_router
from app.routes.sessions import router as sessions_router
from app.routes.system import router as system_router
from app.video_swap import router as video_router

__all__ = ["webhooks_router", "sessions_router", "system_router", "video_router"]
