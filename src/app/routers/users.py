from fastapi import APIRouter, Depends, HTTPException

from app.dependencies import get_user_service
from app.schemas.user import UserResponse
from app.services.user import UserService

router = APIRouter()


@router.get("/{user_id}", response_model=UserResponse)
async def get_user(
    user_id: int,
    service: UserService = Depends(get_user_service),
) -> UserResponse:
    user = service.get_by_id(user_id)
    if user is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user
