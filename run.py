import os
import subprocess
import sys
from pathlib import Path

def main():
    # Установка зависимостей
    print("Устанавливаем зависимости...")
    subprocess.run(["python", "-m", "venv", "venv"])
    
    # Активация venv для Windows и Unix
    if sys.platform == "win32":
        pip_path = "venv/Scripts/pip.exe"
    else:
        pip_path = "venv/bin/pip"
    
    subprocess.run([pip_path, "install", "-r", "requirements.txt"], check=True)
    subprocess.run([pip_path, "install", "-r", "requirements_test.txt"], check=True)
    
    # Создаем необходимые директории
    print("Создаем директории...")
    upload_dir = Path("app") / "static" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # Создаем тестовую директорию для изображений
    test_img_dir = Path("app") / "static" / "test_images"
    test_img_dir.mkdir(parents=True, exist_ok=True)
    
    # Проверяем наличие модели и классов
    model_path = Path("app") / "model_weights.pth"
    classes_path = Path("app") / "classes.txt"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Файл модели не найден: {model_path}")
    
    if not classes_path.exists():
        print(f"Предупреждение: файл классов не найден: {classes_path}")
    
    # ЗАПУСК ТЕСТОВ
    print("Запускаем тесты...")
    test_result = subprocess.run([
        "pytest", "tests", "-v", "--cov=app", "--cov-report=term"
    ])
    
    if test_result.returncode != 0:
        print("\nОШИБКА: Тесты не прошли!")
        print("Приложение не будет запущено до устранения ошибок в тестах")
        sys.exit(1)
    
    # Запускаем приложение
    print("\nТесты успешно пройдены! Запускаем приложение...")
    subprocess.run(["python", "app/main.py"])

if __name__ == "__main__":
    main()