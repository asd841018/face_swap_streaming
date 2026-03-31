"""Faceswap HMAC API Client."""
import hmac
import hashlib
import time
import aiohttp


class FaceswapApiClient:
    """Client for external Faceswap API with HMAC authentication."""

    def __init__(self, base_url: str, api_key: str, api_secret: str):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.api_secret = api_secret

    def _sign(self, method: str, path: str, timestamp: str, content_length: str = "0") -> str:
        msg = f"{method}\n{path}\n{timestamp}\n{content_length}".encode()
        return hmac.new(self.api_secret.encode(), msg, hashlib.sha256).hexdigest()

    def _headers(self, method: str, path: str) -> dict:
        ts = str(int(time.time()))
        return {
            "X-API-Key": self.api_key,
            "X-Signature": self._sign(method, path, ts),
            "X-Timestamp": ts,
        }

    async def get_face_image_async(
        self,
        session: aiohttp.ClientSession,
        timeout_seconds: float = 8.0,
    ) -> dict:
        """Fetch face-image config from the external API (async)."""
        path = "/api/v1/faceswap/face-image"
        url = f"{self.base_url}{path}"
        async with session.get(url, headers=self._headers("GET", path), timeout=timeout_seconds) as resp:
            if resp.status >= 400:
                body = await resp.text()
                raise RuntimeError(f"Faceswap API {resp.status}: {body[:200]}")
            return await resp.json()
