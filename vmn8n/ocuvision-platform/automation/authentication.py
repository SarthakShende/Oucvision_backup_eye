import os
from fastapi import HTTPException, Request, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

security = HTTPBearer()

def get_api_token() -> str:
    token = os.environ.get("OCUVISION_API_TOKEN")
    if not token:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="API token not configured",
        )
    return token

async def verify_token(
    credentials: HTTPAuthorizationCredentials = Depends(security),
    expected: str = Depends(get_api_token),
):
    if credentials.scheme != "Bearer" or credentials.credentials != expected:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing token",
        )
    return True
