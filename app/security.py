from fastapi import Security, HTTPException, status
from fastapi.security import APIKeyHeader
from app.config import API_KEY

API_KEY_NAME = "X-API-Key"
api_key_header = APIKeyHeader(name=API_KEY_NAME, auto_error=False)

def verify_token(api_key: str = Security(api_key_header)):
    if not api_key or api_key != API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Token không hợp lệ hoặc bị thiếu. Vui lòng bấm nút 'Authorize' (hình ổ khóa) ở góc phải trên Swagger UI và nhập API Key."
        )
    return api_key
