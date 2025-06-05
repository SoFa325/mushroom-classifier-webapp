from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from model_loader import MushroomClassifier,load_class_names
import torch
from PIL import Image
from config import Config

def create_app():
    app = Flask(__name__)
    app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'uploads')

    # Создаём папку, если её нет
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    # Ваши настройки и роуты здесь
    @app.route('/')
    def home():
        return render_template('index.html')
        
    @app.route('/guide')
    def guide():
        return render_template('guide.html')

    @app.route('/predict', methods=['GET', 'POST'])  # Разрешаем оба метода
    def predict():
        if request.method == 'POST':
            if 'file' not in request.files:
                return redirect(request.url)
            
            file = request.files['file']
            if file.filename == '':
                return redirect(request.url)
            model_path = os.path.join(app.root_path, 'model_weights.pth')
            if not os.path.exists(model_path):
                return "Model file not found", 500
                
            classifier = MushroomClassifier(
                model_path=model_path,
                class_names=load_class_names()
            )
            if file:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                result = classifier.predict(filepath)
                
                return render_template('predict.html',
                                    image_path=filepath,
                                    prediction=result['predicted_classes'],
                                    probabilities=result['probabilities'])
        
        # GET-запрос: просто отображаем страницу с формой
        return render_template('predict.html')
    
    return app

application = create_app()  # Важно для Amvera!

if __name__ == "__main__":
    application.run(host='0.0.0.0', port=5000)