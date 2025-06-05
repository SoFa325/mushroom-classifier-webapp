import os
import subprocess
from pathlib import Path

def main():
    # Установка зависимостей
    print("Устанавливаем зависимости...")
    print(os.listdir())
    subprocess.run(["python", "-m", "venv", "venv"])
    subprocess.run(["pip", "install", "-r", "requirements.txt"], check=True)
    
    # Создаем необходимые директории
    print("Создаем директории...")
    upload_dir = Path("app") / "static" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # Проверяем наличие модели и классов
    model_path = Path("app") / "model_weights.pth"
    classes_path = Path("app") / "classes.txt"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Файл модели не найден: {model_path}")
    
    if not classes_path.exists():
        print(f"Предупреждение: файл классов не найден: {classes_path}")
    
    """# Запускаем приложение
    print("Запускаем приложение...")
    os.environ["FLASK_APP"] = "app.main:create_app()"
    os.environ["FLASK_ENV"] = "development"
    subprocess.run(["flask", "run", "--host=0.0.0.0", "--port=5000"])"""
    subprocess.run(["python", "app/main.py"])

if __name__ == "__main__":
    main()