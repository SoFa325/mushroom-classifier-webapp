import pytest
import torch
from app.model_loader import MushroomClassifier

@pytest.fixture
def sample_image(tmp_path):
    # Создаём тестовое изображение (можно заменить на реальный пример)
    from PIL import Image
    import numpy as np
    
    img_path = tmp_path / "test_mushroom.jpg"
    img_array = np.random.rand(224, 224, 3) * 255
    img = Image.fromarray(img_array.astype('uint8')).convert('RGB')
    img.save(img_path)
    return img_path

class TestMushroomClassifier:
    @pytest.fixture
    def classifier(self, mocker):
        # Мокаем модель для тестов
        mocker.patch('torch.load')
        mocker.patch('torchvision.models.convnext_base', return_value=torch.nn.Sequential(
            torch.nn.Linear(10, 10)
        ))
        
        return MushroomClassifier(
            model_path="fake_path.pth",
            class_names=["Agaricus", "Amanita"]
        )

    def test_predict(self, classifier, sample_image):
        result = classifier.predict(sample_image)
        assert isinstance(result, dict)
        assert "predicted_classes" in result
        assert "probabilities" in result
        assert len(result["probabilities"]) == len(classifier.class_names)