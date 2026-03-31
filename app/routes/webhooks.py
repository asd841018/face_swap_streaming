from fastapi import APIRouter, Form, Request
from app.services.stream_service import stream_service
from pydantic import BaseModel

router = APIRouter(tags=["Webhooks"])

class WebhookResponse(BaseModel):
    action: str

@router.get("/dev")
async def get_return(request: Request):
    return {"state": "success", "message": "Webhook endpoint is working."}


@router.post("/on_publish", response_model=WebhookResponse)
async def on_publish(
    path: str = Form(...),
    query: str = Form(None)
):
    """
    Webhook triggered by MediaMTX when a stream is published.
    """
    stream_service.start_worker(path, query)
    return {"action": "publish"}


@router.post("/on_publish_done", response_model=WebhookResponse)
async def on_publish_done(
    path: str = Form(...)
):
    """
    Webhook triggered by MediaMTX when a stream ends.
    """
    stream_service.stop_worker(path)
    return {"action": "publish_done"}
