import torch
import pandas as pd
from torchvision import models, transforms
from PIL import Image
from collections import OrderedDict
import os
import json
from pathlib import Path
from mushroom_info import MUSHROOM_SPECIES

class MushroomClassifier:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self._load_model()
        self.class_names = list(MUSHROOM_SPECIES.keys())
        self.transform = self._get_transforms()
    
    def get_russian_name(self, latin_name):
        mushroom = MUSHROOM_SPECIES.get(latin_name, {})
        return mushroom.get('name', latin_name)
    
    def is_edible(self, latin_name):
        mushroom = MUSHROOM_SPECIES.get(latin_name, {})
        return mushroom.get('edible', False)

    def _load_model(self):
        """Загрузка модели с весами"""
        model_path = os.path.join(os.path.dirname(__file__), '../model_weights.pth')
        
        # Загрузка state_dict с обработкой DataParallel
        state_dict = torch.load(model_path, map_location=self.device)
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k.replace('module.', '')
            new_state_dict[name] = v

        # Инициализация модели
        model = models.convnext_base(pretrained=False)
        model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, len(MUSHROOM_SPECIES))
        model.load_state_dict(new_state_dict)
        model.to(self.device)
        model.eval()
        
        return model

    def _get_transforms(self):
        """Трансформации для изображений"""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def predict_image(self, image_path, top_k=2):
        """Предсказание с возвратом топ-k классов"""
        try:
            image = Image.open(image_path).convert('RGB')
            input_tensor = self.transform(image).unsqueeze(0).to(self.device)

            with torch.no_grad():
                output = self.model(input_tensor)
                probs = torch.sigmoid(output).cpu().numpy().flatten()

            top_indices = probs.argsort()[-top_k:][::-1]
            return [
                {"class": self.get_russian_name(self.class_names[i]), "probability": float(probs[i]), "edible": self.is_edible(self.class_names[i])} 
                for i in top_indices
            ]
        except Exception as e:
            print(f"Prediction error: {e}")
            return []