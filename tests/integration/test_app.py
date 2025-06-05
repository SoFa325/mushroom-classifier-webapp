import pytest
import os
from pathlib import Path
from app.main import create_app
from werkzeug.datastructures import FileStorage

# Фикстура приложения с реальной конфигурацией
@pytest.fixture
def app():
    app = create_app()
    app.config.update({
        "TESTING": True,
        "UPLOAD_FOLDER": Path(__file__).parent.parent.parent / "app" / "static" / "uploads"
    })
    yield app

@pytest.fixture
def client(app):
    return app.test_client()

@pytest.fixture
def real_mushroom_image():
    root = Path(__file__).parent.parent.parent
    return open(root / "app" / "static" / "test_images" / "test_mushroom.jpg", "rb")

def test_home_page(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"Грибной определитель" in response.data

def test_guide_page(client):
    response = client.get("/guide")
    assert response.status_code == 200
    assert b"Руководство" in response.data

def test_predict_page_get(client):
    response = client.get("/predict")
    assert response.status_code == 200
    assert b"Определить гриб" in response.data

def test_predict_image_upload(client, real_mushroom_image):
    """Тест загрузки реального изображения"""
    response = client.post("/predict", data={
        "file": (real_mushroom_image, "test_mushroom.jpg")
    }, content_type="multipart/form-data")
    
    assert response.status_code == 200
    assert b"Результаты распознавания" in response.data
    assert b"class" in response.data
    assert b"probability" in response.data

def test_invalid_file_upload(client):
    """Тест загрузки невалидного файла"""
    invalid_file = (BytesIO(b"Not an image"), "invalid.txt")
    response = client.post("/predict", data={
        "file": invalid_file
    }, content_type="multipart/form-data")
    
    assert response.status_code == 200
    assert b"Не удалось обработать изображение" in response.data

def test_no_file_upload(client):
    """Тест запроса без файла"""
    response = client.post("/predict")
    assert response.status_code == 302  # Редирект обратно на страницу