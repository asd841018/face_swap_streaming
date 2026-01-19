"""
Faceswap HMAC API Client
"""
import hmac
import hashlib
import time
import requests
from typing import Optional


class FaceswapApiClient:
    """
    Faceswap API 客戶端，使用 HMAC 認證
    
    用於與外部 Faceswap API 通訊，獲取用戶的換臉來源圖片等資訊。
    """
    
    def __init__(self, base_url: str, api_key: str, api_secret: str):
        """
        初始化客戶端
        
        Args:
            base_url: API 基底 URL (例如: https://api.aimate.am)
            api_key: 使用者的 API Key
            api_secret: 使用者的 API Secret
        """
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.api_secret = api_secret
    
    def _generate_signature(
        self, 
        method: str, 
        path: str, 
        timestamp: str, 
        content_length: str = "0"
    ) -> str:
        """
        產生 HMAC-SHA256 簽名
        
        簽名格式: {METHOD}\n{PATH}\n{TIMESTAMP}\n{CONTENT_LENGTH}
        """
        message = f"{method}\n{path}\n{timestamp}\n{content_length}".encode()
        signature = hmac.new(
            self.api_secret.encode(),
            message,
            hashlib.sha256
        ).hexdigest()
        return signature
    
    def _get_headers(self, method: str, path: str, content_length: int = 0) -> dict:
        """產生帶有 HMAC 認證的 headers"""
        timestamp = str(int(time.time()))
        signature = self._generate_signature(
            method=method,
            path=path,
            timestamp=timestamp,
            content_length=str(content_length)
        )
        
        return {
            "X-API-Key": self.api_key,
            "X-Signature": signature,
            "X-Timestamp": timestamp,
        }
   
    def get_face_image(self) -> dict:
        """
        取得臉部圖片 URL
        
        Returns:
            API 回應 dict，格式:
            {
                "code": 0,
                "message": "ok",
                "data": {
                    "face_image_url": "https://...",
                    "user_id": "uuid-string"
                }
            }
        """
        path = "/api/v1/faceswap/face-image"
        url = f"{self.base_url}{path}"
        
        headers = self._get_headers("GET", path)
        
        response = requests.get(url, headers=headers)
        return response.json()


if __name__ == "__main__":
    # 範例使用
    client = FaceswapApiClient(
        base_url="https://api.aimate.am",
        api_key="ak_am_xyNgXgGFYmn3BcWICcHZDoEVM027Kdu9",
        api_secret="E5E2yQFH3G7D18IisPKiNYYsYbl5FEZK"
    )
    response = client.get_face_image()
    print(response)
