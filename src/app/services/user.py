from app.models.user import User

FAKE_USERS = {
    1: User(id=1, name="Alice", email="alice@example.com"),
    2: User(id=2, name="Bob", email="bob@example.com"),
}


class UserService:
    def get_by_id(self, user_id: int) -> User | None:
        return FAKE_USERS.get(user_id)
