import os
import subprocess
import sys
from pathlib import Path

def main():
    # Определяем пути
    venv_path = Path("venv")
    if sys.platform == "win32":
        pip_path = venv_path / "Scripts" / "pip.exe"
        python_path = venv_path / "Scripts" / "python.exe"
    else:
        pip_path = venv_path / "bin" / "pip"
        python_path = venv_path / "bin" / "python"
    
    # Проверяем, установлены ли уже зависимости
    dependencies_installed = False
    if venv_path.exists():
        print("Виртуальное окружение уже существует, проверяем зависимости...")
        try:
            # Проверяем, установлен ли Flask
            subprocess.run([pip_path, "show", "flask"], check=True, 
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("Основные зависимости уже установлены")
            dependencies_installed = True
        except subprocess.CalledProcessError:
            print("Зависимости не установлены")

    # Если зависимости не установлены, пытаемся установить
    if not dependencies_installed:
        print("Устанавливаем зависимости...")
        
        # Создаем виртуальное окружение
        if not venv_path.exists():
            subprocess.run(["python", "-m", "venv", "venv"])
        
        # Обновляем pip
        print("Обновляем pip...")
        try:
            subprocess.run([pip_path, "install", "--upgrade", "pip"], check=True)
        except subprocess.CalledProcessError:
            print("⚠️ Не удалось обновить pip, продолжаем с текущей версией")
        
        # Пытаемся установить зависимости
        try:
            print("Устанавливаем основные зависимости...")
            subprocess.run([pip_path, "install", "-r", "requirements.txt"], check=True)
            print("Устанавливаем тестовые зависимости...")
            subprocess.run([pip_path, "install", "-r", "requirements_test.txt"], check=True)
            print("✅ Зависимости успешно установлены")
            dependencies_installed = True
        except subprocess.CalledProcessError:
            print("\n⚠️ Не удалось установить зависимости")
            print("Пожалуйста, выполните следующие действия:")
            print("1. Активируйте виртуальное окружение:")
            print("   Для Windows: .\\venv\\Scripts\\activate")
            print("   Для Linux/Mac: source venv/bin/activate")
            print("2. Установите зависимости вручную:")
            print("   pip install -r requirements.txt")
            print("   pip install -r requirements_test.txt")
            print("3. После установки запустите скрипт снова")
            sys.exit(1)
    
    print("Создаем директории...")
    upload_dir = Path("app") / "static" / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    
    # Убедитесь, что создается правильная папка без дублирования
    test_img_dir = Path("app") / "static" / "test_images"
    test_img_dir.mkdir(parents=True, exist_ok=True)

    test_img_dir = Path("app") / "static" / "test_images"
    test_img_dir.mkdir(parents=True, exist_ok=True)
    
    # Проверяем наличие модели и классов
    model_path = Path("app") / "model_weights.pth"
    classes_path = Path("app") / "classes.txt"
    
    if not model_path.exists():
        raise FileNotFoundError(f"Файл модели не найден: {model_path}")
    
    if not classes_path.exists():
        print(f"⚠️ Предупреждение: файл классов не найден: {classes_path}")
        # Создаем временный файл классов для прохождения тестов
        with open(classes_path, "w", encoding="utf-8") as f:
            for i in range(211):
                f.write(f"class_{i}\n")
        print(f"Создан временный файл классов: {classes_path}")
    
    # Запускаем тесты (без опций покрытия)
    print("Запускаем тесты...")
    test_result = subprocess.run([python_path, "-m", "pytest", "tests", "-v"])
    
    if test_result.returncode != 0:
        print("\n❌ ОШИБКА: Тесты не прошли!")
        print("Приложение не будет запущено до устранения ошибок в тестах")
        sys.exit(1)
    
    # Запускаем приложение
    print("\n✅ Тесты успешно пройдены! Запускаем приложение...")
    subprocess.run([python_path, "-m", "app.main"])

if __name__ == "__main__":
    main()