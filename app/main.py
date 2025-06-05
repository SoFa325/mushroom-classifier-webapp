from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from model_loader import MushroomClassifier
import torch
from PIL import Image
from config import Config
from model_loader import load_class_names

def create_app():
    app = Flask(__name__)
    
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
            
            classifier = MushroomClassifier(
                model_path="model_weights.pth",
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