import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
import os
from collections import OrderedDict

import pandas as pd


map_location=torch.device('cpu')
state_dict = torch.load('C:/Users/sofya/IdeaProjects/mushroom_project/mushroom-classifier-webapp/app/model_weights.pth', map_location)
# Удаляем 'module.' из ключей, если они есть

new_state_dict = OrderedDict()
for k, v in state_dict.items():
    name = k.replace('module.', '')  # удаляем 'module.'
    new_state_dict[name] = v

model = models.convnext_base(pretrained=False)
model.classifier[2] = torch.nn.Linear(model.classifier[2].in_features, 211)
model.load_state_dict(new_state_dict)
model.to('cpu')
model.eval()

# Загрузка CSV-файла
df = pd.read_csv(r'C:\Users\sofya\Downloads\Mantar_tanima.v2i.multiclass(1)\train\_classes.csv')

# Получение списка имён классов (все столбцы, кроме первого)
class_names = df.columns[1:].tolist()

import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from torchvision import transforms

def predict_image_top5(image_path, model, class_names, device, top_k=5):
    # Загрузка и подготовка изображения
    image = Image.open(image_path).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    input_tensor = transform(image).unsqueeze(0).to(device)

    # Предсказание
    model.eval()
    with torch.no_grad():
        output = model(input_tensor)
        probs = torch.sigmoid(output).cpu().numpy().flatten()

    # Получаем индексы топ-k вероятностей
    top_indices = probs.argsort()[-top_k:][::-1]
    top_classes = [(class_names[i], probs[i]) for i in top_indices]
    print(top_classes)
    # Вывод изображения и топ-5 классов с вероятностями
    plt.figure(figsize=(8, 6))
    plt.imshow(image)
    plt.axis('off')
    title = "Top 5 predictions:\n" + "\n".join([f"{cls}: {prob:.4f}" for cls, prob in top_classes])
    plt.title(title, fontsize=12)
    plt.show()

    # Возвращаем список топ-5 классов и вероятностей
    return top_classes

# Пример использования:
image_path = "mushroom-classifier-webapp/app/uploads/Agaricus_arvensis12_png.rf.160ae54d3c8b317fba16c4050b4e87a7.jpg"
top5 = predict_image_top5(image_path, model, class_names, device='cpu')
print(top5)