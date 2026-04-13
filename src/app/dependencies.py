from app.services.user import UserService


def get_user_service() -> UserService:
    return UserService()
