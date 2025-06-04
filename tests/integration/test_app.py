import pytest
from app.main import app as flask_app

@pytest.fixture
def app():
    flask_app.config.update({
        "TESTING": True,
    })
    yield flask_app

@pytest.fixture
def client(app):
    return app.test_client()

def test_home_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Грибной определитель" in response.data

def test_predict_page(client):
    response = client.get("/predict")
    assert response.status_code == 200
    assert b"Определить гриб" in response.data