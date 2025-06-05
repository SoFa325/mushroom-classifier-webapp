import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os

def load_class_names(file_path="app/classes.txt"):
    """Загружает список классов из текстового файла"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            class_names = [line.strip() for line in f if line.strip()]
        return class_names
    except FileNotFoundError:
        print(f"Warning: Class names file not found at {file_path}, using defaults")
        return []

class MushroomClassifier:
    def __init__(self, model_path, class_file="app/classes.txt"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.class_names = load_class_names(class_file)
        self.num_classes = len(self.class_names)
        
        # Инициализация модели (как в оригинальном обучении)
        self.model = models.convnext_base(pretrained=False)
        self.model.classifier[2] = nn.Linear(self.model.classifier[2].in_features, self.num_classes)
        
        # Загрузка весов
        self.load_model(model_path)
        self.model.to(self.device)
        self.model.eval()
        
        # Трансформации (как в оригинальном обучении)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def load_model(self, model_path):
        """Загрузка сохраненных весов модели"""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model weights file not found at {model_path}")
        
        # Для совместимости с DataParallel (если использовалось)
        state_dict = torch.load(model_path, map_location=self.device)
        if 'module' in list(state_dict.keys())[0]:
            # Удаляем 'module.' из ключей, если модель была сохранена с DataParallel
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                name = k[7:]  # удаляем 'module.'
                new_state_dict[name] = v
            state_dict = new_state_dict
        
        self.model.load_state_dict(state_dict)

    def predict(self, image_path, threshold=0.5):
        """Предсказание классов для одного изображения"""
        try:
            image = Image.open(image_path).convert('RGB')
            image = self.transform(image).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                outputs = self.model(image)
                probs = torch.sigmoid(outputs)  # Для multi-label classification
                preds = (probs > threshold).cpu().numpy().flatten()
            
            # Формируем результат
            result = {
                'predicted_classes': [],
                'probabilities': {cls: float(prob) for cls, prob in zip(self.class_names, probs.cpu().numpy().flatten())}
            }
            
            for i, is_present in enumerate(preds):
                if is_present:
                    result['predicted_classes'].append(self.class_names[i])
            
            return result
            
        except Exception as e:
            print(f"Error during prediction: {e}")
            return None

# Пример использования:
if __name__ == "__main__":
    # Пример списка классов (должен соответствовать обученной модели)
    example_class_names = [
        "Agaricus", "Amanita", "Boletus", "Cortinarius", 
        "Entoloma", "Hygrocybe", "Lactarius", "Russula", "Suillus"
    ]
    
    classifier = MushroomClassifier(
        model_path="model_weights.pth",
        class_names=example_class_names
    )
    
    # Тестовый пример
    test_image = "test_mushroom.jpg"
    if os.path.exists(test_image):
        prediction = classifier.predict(test_image)
        print("Prediction results:")
        print(f"Detected classes: {prediction['predicted_classes']}")
        print("Probabilities:")
        for cls, prob in prediction['probabilities'].items():
            print(f"{cls}: {prob:.4f}")
    else:
        print(f"Test image {test_image} not found")