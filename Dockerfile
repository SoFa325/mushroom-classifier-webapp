# Базовый образ
FROM python:3.9-slim

# Установка зависимостей
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем зависимости
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем проект
COPY . .
COPY app/model/model_loader.py /app/app/model/model_loader.py
COPY app/model_weights.pth /app/app/model_weights.pth

# Запуск приложения
CMD gunicorn --bind 0.0.0.0:$PORT --workers 4 --threads 2 app.main:app