#!/bin/bash
git clone https://github.com/SoFa325/mushroom-classifier-webapp.git
cd mushroom-classifier

# Создаем виртуальное окружение
python -m venv venv
# Установка Python зависимостей
pip install -r requirements.txt