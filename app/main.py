import uvicorn
import asyncio
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.services.monitor import monitor_streams
from app.core import logger
from app.config import settings
from app.routes.webhooks import router as webhooks_router
from app.routes.sessions import router as sessions_router
from app.routes.system import router as system_router
from app.routes.video import router as video_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("[Main] Server starting...")
    # Start the background monitor task
    task = asyncio.create_task(monitor_streams())
    yield
    logger.info("[Main] Server shutting down...")
    task.cancel()

app = FastAPI(title="AI RTMP Stream Manager", 
              description="Manage AI RTMP streams with ease",
              version="1.0.0",
              lifespan=lifespan)
# Added this block
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # Allows all origins
    allow_credentials=True,
    allow_methods=["*"], # Allows all methods
    allow_headers=["*"], # Allows all headers
)

app.include_router(webhooks_router)
app.include_router(sessions_router)
app.include_router(system_router)
app.include_router(video_router)

if __name__ == "__main__":
    # Run from the root directory: python -m app.main
    uvicorn.run(app, 
                host=settings.HOST, 
                port=settings.PORT,
            )
