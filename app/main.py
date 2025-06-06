from flask import Flask, render_template, request, redirect, url_for
from werkzeug.utils import secure_filename
import os
from app.model.model_loader import MushroomClassifier

def create_app():
    app = Flask(__name__)
    app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'static',"uploads")
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
            classifier = MushroomClassifier()
            file = request.files['file']
            if file.filename == '':
                return redirect(request.url)
            
            if file:
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'],filename)
                file.save(filepath)
                
                # Получаем предсказания
                predictions = classifier.predict_image(filepath)
                
                return render_template('predict.html',
                                    image_path=filepath,
                                    predictions=predictions)
        # GET-запрос: просто отображаем страницу с формой
        return render_template('predict.html')
    
    return app

application = create_app()  # Важно для Amvera!

if __name__ == "__main__":
    application.run(host='0.0.0.0', port=5000)