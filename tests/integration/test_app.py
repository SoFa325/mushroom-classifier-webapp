import pytest
import os
from pathlib import Path
from io import BytesIO
from app.main import create_app
from werkzeug.datastructures import FileStorage

# Фикстура приложения с реальной конфигурацией
@pytest.fixture
def app():
    app = create_app()
    upload_folder = Path(__file__).parent.parent.parent / "app" / "static" / "uploads"
    upload_folder.mkdir(parents=True, exist_ok=True)  # Создаем папку для загрузок
    
    app.config.update({
        "TESTING": True,
        "UPLOAD_FOLDER": upload_folder
    })
    yield app

    # Очистка после тестов
    for file in upload_folder.glob("*"):
        if file.is_file():
            file.unlink()

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
    assert "Грибник.ру".encode('utf-8') in response.data 

def test_guide_page(client):
    response = client.get("/guide")
    assert response.status_code == 200
    assert "Справочник".encode('utf-8') in response.data  

def test_predict_page_get(client):
    response = client.get("/predict")
    assert response.status_code == 200
    assert "Определить гриб".encode('utf-8') in response.data  

def test_predict_image_upload(client, real_mushroom_image):
    response = client.post("/predict", data={
        "file": (real_mushroom_image, "test_mushroom.jpg")
    }, content_type="multipart/form-data")
    
    assert response.status_code == 200
    assert "Результаты".encode('utf-8') in response.data  
    assert "class".encode('utf-8') in response.data
    assert "probability".encode('utf-8') in response.data


def test_no_file_upload(client):
    response = client.post("/predict")
    assert response.status_code == 302  # Редирект обратно на страницу