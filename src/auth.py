from fastapi import HTTPException, Header, status
from src.supabase_client import get_supabase_client


async def get_current_user(authorization: str = Header(None)) -> str:
    """
    Validate Supabase JWT token and return user_id.

    Frontend sends: Authorization: Bearer <supabase_jwt_token>
    Backend verifies with Supabase and extracts user_id.
    """
    if not authorization:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing Authorization header. Provide Bearer token.",
        )

    # Extract token from "Bearer <token>"
    parts = authorization.split(" ")
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid Authorization header. Use: Bearer <token>",
        )

    token = parts[1]

    # Verify token with Supabase
    try:
        supabase = get_supabase_client()
        user_response = supabase.auth.get_user(token)

        if not user_response or not user_response.user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid or expired token",
            )

        return str(user_response.user.id)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=f"Token validation failed: {str(e)}",
        )
