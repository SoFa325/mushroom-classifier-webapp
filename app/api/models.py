import torch
import torch.nn as nn
from torchvision import models
from PIL import Image
from torchvision import transforms
import os

class MushroomClassifier:
    """Класс для загрузки модели и выполнения предсказаний"""
    
    def __init__(self, model_path: str, class_names: list):
        """
        Args:
            model_path: Путь к файлу весов модели (.pth)
            class_names: Список названий классов грибов
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = class_names
        self.model = self._load_model(model_path)
        self.transform = self._get_transforms()

    def _load_model(self, model_path: str) -> nn.Module:
        """Загружает предобученную модель"""
        model = models.convnext_base(pretrained=False)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, len(self.class_names))
        
        # Загрузка весов
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights not found at {model_path}")
        
        state_dict = torch.load(model_path, map_location=self.device)
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    def _get_transforms(self):
        """Возвращает трансформации для изображений"""
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict(self, image_path: str, threshold: float = 0.5) -> dict:
        """Выполняет предсказание на изображении"""
        try:
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(image)
                probs = torch.sigmoid(outputs)  # Для multi-label классификации
                preds = (probs > threshold).cpu().numpy().flatten()
            
            return {
                "predicted_classes": [self.class_names[i] for i, pred in enumerate(preds) if pred],
                "probabilities": {cls: float(prob) for cls, prob in zip(self.class_names, probs.cpu().numpy().flatten())}
            }
        except Exception as e:
            return {"error": str(e)}