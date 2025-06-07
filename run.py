import subprocess
import sys
from pathlib import Path

def main():

    venv_python = Path("venv") / "Scripts" / "python.exe" if sys.platform == "win32" else Path("venv") / "bin" / "python"
    
    (Path("app") / "static" / "uploads").mkdir(parents=True, exist_ok=True)
    (Path("app") / "static" / "test_images").mkdir(exist_ok=True)

    if "--skip-tests" not in sys.argv:
        subprocess.run([venv_python, "-m", "pytest", "tests/", "-v"], check=True)

    subprocess.run([venv_python, "-m", "app.main"])

if __name__ == "__main__":
    main()