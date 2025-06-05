import torch
import torch.nn as nn
from torchvision import models

class MushroomClassifier:
    """Класс для загрузки модели и выполнения предсказаний"""
    
    def __init__(self, model_path: str, class_names: list):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.class_names = class_names
        self.model = self._load_model(model_path)
        self.transform = self._get_transforms()
    
        # Проверка соответствия числа классов
        current_classes = self.model.classifier[2].out_features
        if current_classes != len(class_names):
            print(f"Предупреждение: Модель обучена на {current_classes} классов, а передано {len(class_names)}")

    def _load_model(self, model_path: str) -> nn.Module:
        checkpoint = torch.load(model_path, map_location=self.device)
        print("Содержимое checkpoint:")
        for key in list(checkpoint.keys())[:5]:  # Выводим первые 5 ключей
            print(f"{key}: {checkpoint[key].shape if hasattr(checkpoint[key], 'shape') else type(checkpoint[key])}")
        
        # Определяем число классов из размерности последнего слоя
        if 'module.classifier.2.weight' in checkpoint:
            num_classes = checkpoint['module.classifier.2.weight'].size(0)
        elif 'classifier.2.weight' in checkpoint:
            num_classes = checkpoint['classifier.2.weight'].size(0)
        else:
            num_classes = len(self.class_names)  # Используем длину переданного списка классов
        
        # Инициализируем модель
        model = models.convnext_base(pretrained=False)
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)
        
        # Обработка DataParallel (удаляем 'module.' из ключей)
        state_dict = {}
        for k, v in checkpoint.items():
            if k.startswith('module.'):
                state_dict[k[7:]] = v
            else:
                state_dict[k] = v
        
        # Загружаем веса
        model.load_state_dict(state_dict)
        model.to(self.device)
        model.eval()
        return model

    def _get_transforms(self):
        from torchvision import transforms
        return transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict(self, image_path: str, threshold: float = 0.5) -> dict:
        from PIL import Image
        try:
            image = Image.open(image_path).convert("RGB")
            image = self.transform(image).unsqueeze(0).to(self.device)
            with torch.no_grad():
                outputs = self.model(image)
                probs = torch.sigmoid(outputs)
                preds = (probs > threshold).cpu().numpy().flatten()
            return {
                "predicted_classes": [self.class_names[i] for i, pred in enumerate(preds) if pred],
                "probabilities": {cls: float(prob) for cls, prob in zip(self.class_names, probs.cpu().numpy().flatten())}
            }
        except Exception as e:
            return {"error": str(e)}
