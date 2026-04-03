from fastapi import HTTPException, Header, status
from config.settings import Settings


async def authorize_request(x_api_key: str = Header(None)):
    """Validate API key from X-API-Key header."""
    if not Settings.ACCESS_API_KEY:
        return  # No key configured — skip auth (dev mode)

    if not x_api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key. Provide X-API-Key header.",
        )

    if x_api_key != Settings.ACCESS_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key",
        )
