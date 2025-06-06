import pytest
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from app.model.model_loader import MushroomClassifier

# Фикстура с реальным классификатором
@pytest.fixture(scope="module")
def real_classifier():
    # Пути относительно корня проекта
    root = Path(__file__).parent.parent.parent
    return MushroomClassifier()

# Фикстура с тестовым изображением
@pytest.fixture
def real_mushroom_image():
    root = Path(__file__).parent.parent.parent
    return root / "app" / "static" / "test_images" / "test_mushroom.jpg"

def test_model_loading(real_classifier):
    """Проверка загрузки модели"""
    assert real_classifier.model is not None
    assert isinstance(real_classifier.model, torch.nn.Module)

def test_class_names_loading(real_classifier):
    """Проверка загрузки классов"""
    assert len(real_classifier.class_names) > 0
    assert all(isinstance(name, str) for name in real_classifier.class_names)

def test_prediction_on_real_image(real_classifier, real_mushroom_image):
    """Тест предсказания на реальном изображении"""
    results = real_classifier.predict_image(real_mushroom_image)
    assert len(results) > 0
    for item in results:
        assert "class" in item
        assert "probability" in item
        assert 0 <= item["probability"] <= 1

def test_prediction_failure(real_classifier):
    """Тест обработки невалидного изображения"""
    invalid_image = Path(__file__).parent / "invalid_image.txt"
    with open(invalid_image, "w") as f:
        f.write("This is not an image")
    
    results = real_classifier.predict_image(invalid_image)
    assert len(results) == 0

def test_prediction_format(real_classifier, real_mushroom_image):
    """Тест формата выходных данных"""
    results = real_classifier.predict_image(real_mushroom_image, top_k=3)
    assert len(results) == 3
    assert results[0]["probability"] >= results[1]["probability"]

def test_image_types(real_classifier):
    """Тест работы с разными типами изображений"""
    image_formats = ["JPEG", "PNG", "BMP"]
    for fmt in image_formats:
        img_path = Path(__file__).parent / f"test_image.{fmt.lower()}"
        img = Image.new("RGB", (224, 224), color="red")
        img.save(img_path, format=fmt)
        
        results = real_classifier.predict_image(img_path)
        assert len(results) > 0
        img_path.unlink()