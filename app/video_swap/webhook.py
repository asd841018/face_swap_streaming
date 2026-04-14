"""Outbound webhook delivery for video job progress updates.

Fire-and-forget: delivery failures are retried but never propagate to the job.
Optionally signs the request body with HMAC-SHA256 so the receiver can verify
authenticity.
"""
import asyncio
import hashlib
import hmac
import json
from typing import Any, Dict

import httpx

from app.config import settings
from app.core import logger


class WebhookDelivery:
    def __init__(self) -> None:
        self._timeout = settings.WEBHOOK_TIMEOUT_SECONDS
        self._max_retries = max(1, settings.WEBHOOK_MAX_RETRIES)
        self._secret = settings.WEBHOOK_SIGNING_SECRET

    def _sign(self, body: bytes) -> str:
        return hmac.new(self._secret.encode(), body, hashlib.sha256).hexdigest()

    async def _deliver(self, url: str, payload: Dict[str, Any]) -> None:
        body = json.dumps(payload, default=str).encode()
        headers = {
            "Content-Type": "application/json",
            "User-Agent": "face-swap-streaming/1.0",
            "X-Webhook-Event": payload.get("event", "job.updated"),
            "X-Webhook-Job-Id": str(payload.get("job_id", "")),
        }
        if self._secret:
            headers["X-Webhook-Signature"] = f"sha256={self._sign(body)}"

        backoff = 1.0
        for attempt in range(1, self._max_retries + 1):
            try:
                async with httpx.AsyncClient(timeout=self._timeout) as client:
                    response = await client.post(url, content=body, headers=headers)
                if 200 <= response.status_code < 300:
                    logger.info(
                        f"[Webhook] Delivered job={payload.get('job_id')} "
                        f"event={payload.get('event')} → {url} ({response.status_code})"
                    )
                    return
                logger.warning(
                    f"[Webhook] {url} returned {response.status_code} (attempt {attempt}/{self._max_retries})"
                )
            except Exception as e:
                logger.warning(f"[Webhook] {url} attempt {attempt}/{self._max_retries} failed: {e}")

            if attempt < self._max_retries:
                await asyncio.sleep(backoff)
                backoff *= 2

        logger.error(
            f"[Webhook] Giving up on {url} after {self._max_retries} attempts "
            f"(job={payload.get('job_id')}, event={payload.get('event')})"
        )

    def schedule(self, url: str, payload: Dict[str, Any]) -> None:
        """Fire-and-forget delivery scheduled on the running event loop."""
        task = asyncio.create_task(self._deliver(url, payload))
        task.add_done_callback(lambda t: t.exception())


webhook_delivery = WebhookDelivery()
