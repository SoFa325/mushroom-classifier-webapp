import subprocess
import sys
from pathlib import Path

def main():
    project_dir = Path(__file__).parent.resolve()
    venv_dir = project_dir / "venv"

    try:
        venv_python = venv_dir / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")
        venv_python = venv_python.resolve(strict=True)
    except FileNotFoundError:
        print(f"""
❌ Ошибка: Виртуальное окружение не найдено по пути:
{venv_dir}

🔍 Решение:
1. Создайте виртуальное окружение в пути без кириллицы:
   python -m venv C:\\projects\\mushroom_app\\venv
2. Активируйте его:
   C:\\projects\\mushroom_app\\venv\\Scripts\\activate
3. Переустановите зависимости:
   pip install -r requirements.txt
""")
        sys.exit(1)

    # Установка основных зависимостей
    try:
        subprocess.run(
            [str(venv_python), "-m", "pip", "install", "-r", "requirements.txt"],
            check=True,
            cwd=project_dir
        )
        subprocess.run(
            [str(venv_python), "-m", "pip", "install", "-r", "requirements_test.txt"],
            check=True,
            cwd=project_dir
        )
    except subprocess.CalledProcessError as e:
        print(f"❌ Ошибка установки зависимостей: {e}")
        sys.exit(1)

    # Создание директорий
    uploads_dir = project_dir / "app/static/uploads"
    test_images_dir = project_dir / "app/static/test_images"
    
    uploads_dir.mkdir(parents=True, exist_ok=True)
    test_images_dir.mkdir(exist_ok=True)

    # Запуск тестов
    if "--skip-tests" not in sys.argv:
        try:
            test_cmd = [
                str(venv_python), "-m", "pytest",
                "tests/", "-v",
                "--cov=app",
                "--cov-report=term-missing"
            ]
            subprocess.run(test_cmd, check=True, cwd=project_dir)
        except subprocess.CalledProcessError as e:
            print(f"""
❌ Тесты не пройдены (код ошибки: {e.returncode})
🔍 Проверьте:
1. Наличие файлов Flask и Torch в venv
2. Корректность путей в импортах
3. Содержимое requirements-файлов
""")
            sys.exit(e.returncode)

    # Запуск приложения
    subprocess.run(
        [str(venv_python), "-m", "app.main"],
        cwd=project_dir
    )

if __name__ == "__main__":
    main()

