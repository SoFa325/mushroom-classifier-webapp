import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import os
from api.models import MushroomClassifier

def load_class_names(file_path="classes.txt"):
    """Загружает список классов из текстового файла"""
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            class_names = [line.strip() for line in f if line.strip()]
        return class_names
    except FileNotFoundError:
        print(f"Warning: Class names file not found at {file_path}, using defaults")
        return []

# Пример использования:
if __name__ == "__main__":
    # Пример списка классов (должен соответствовать обученной модели)
    class_names=load_class_names()
    assert len(class_names) == 211, f"Ожидается 211 классов, получено {len(class_names)}"
    classifier = MushroomClassifier(
        model_path="app/model_weights.pth",
        class_names = class_names 
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