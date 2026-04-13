from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def test_get_user() -> None:
    response = client.get("/users/1")
    assert response.status_code == 200
    assert response.json() == {
        "id": 1,
        "name": "Alice",
        "email": "alice@example.com",
    }


def test_get_user_not_found() -> None:
    response = client.get("/users/999")
    assert response.status_code == 404
