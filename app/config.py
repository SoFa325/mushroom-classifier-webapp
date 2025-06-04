import os
from dotenv import load_dotenv

# Загрузка переменных из .env
load_dotenv()

class Config:
    # Общие настройки
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key')
    MODEL_PATH = os.getenv('MODEL_PATH', 'app/model_weights.pth')
    UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'static/uploads')
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
    
    # Настройки для Amvera
    PORT = int(os.getenv('PORT', 8000))
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16 MB

class DevelopmentConfig(Config):
    DEBUG = True
    FLASK_ENV = 'development'

class ProductionConfig(Config):
    DEBUG = False
    FLASK_ENV = 'production'

# Выбор конфига по окружению
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}