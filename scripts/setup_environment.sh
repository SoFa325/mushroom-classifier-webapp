#!/bin/bash
git clone https://github.com/SoFa325/mushroom-classifier-webapp.git
cd mushroom-classifier

# Создаем виртуальное окружение
python -m venv venv
# Установка Python зависимостей
pip install -r requirements.txt

# Скачивание модели (пример)
if [ ! -f "app/model_weights.pth" ]; then
    wget -O app/model_weights.pth "https://your-model-storage.com/model_weights.pth"
fi

# Создание необходимых директорий
mkdir -p app/static/uploads