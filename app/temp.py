# В временном скрипте
import torch
data = torch.load('C:/Users/sofya/IdeaProjects/mushroom_project/mushroom-classifier-webapp/app/model_weights.pth')
print("Ключи в файле:", data.keys())
print("Форма последнего слоя:", data['classifier.2.weight'].shape)