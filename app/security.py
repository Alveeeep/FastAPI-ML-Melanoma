from __future__ import annotations

from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.settings import Settings

bearer = HTTPBearer(auto_error=False)


def auth_dependency(settings: Settings):
    def _check(credentials: HTTPAuthorizationCredentials | None = Depends(bearer)) -> None:
        if not settings.enable_auth:
            return
        if credentials is None or credentials.scheme.lower() != "bearer":
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Missing bearer token.")
        if credentials.credentials != settings.api_token:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Invalid bearer token.")

    return _check

