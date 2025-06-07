from flask import Flask, render_template, jsonify, request, redirect, url_for
import requests
from werkzeug.utils import secure_filename
from bs4 import BeautifulSoup
import os
from urllib.parse import urljoin
import hashlib
from mushroom_info import MUSHROOM_SPECIES
from app.model.model_loader import MushroomClassifier

def create_app():
    app = Flask(__name__)
    app.config['UPLOAD_FOLDER'] = os.path.join(app.root_path, 'static',"uploads")
    app.config['WIKI_IMAGES'] = os.path.join(app.root_path, 'static', "wiki_images")
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['WIKI_IMAGES'], exist_ok=True)

    def get_wiki_description(wiki_url):
        try:
            response = requests.get(wiki_url, timeout=15)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Ищем основной контент
            content = soup.find('div', {'class': 'mw-parser-output'})
            if not content:
                return "Не удалось загрузить описание"
            
            # Удаляем навигационные и служебные элементы (НЕ все div!)
            for element in content.find_all(['table', 'span', 'img', 'sup', '.navbox', '.infobox']):
                element.decompose()
            
            # Удаляем div с классами навбоксов и инфобоксов
            for element in content.find_all('div', class_=['navbox', 'infobox', 'hatnote', 'dablink']):
                element.decompose()
            
            # Берем первые 3-4 осмысленных абзаца
            paragraphs = []
            for p in content.find_all('p'):
                text = p.get_text().strip()
                # Фильтруем абзацы с координатами и служебной информацией
                if (text and len(text) > 30 and 
                    not text.startswith('Координаты:') and
                    not text.startswith('Commons:') and
                    '°' not in text[:50]):  # Исключаем координаты
                    paragraphs.append(text)
                    if len(paragraphs) >= 3:
                        break
            
            result = ' '.join(paragraphs)
            return result[:800] + '...' if len(result) > 800 else result or "Описание отсутствует"
        
        except Exception as e:
            print(f"Error fetching description from {wiki_url}: {e}")
            return "Не удалось загрузить описание"

    @app.route('/')
    def home():
        return render_template('index.html')

    @app.route('/guide')
    def guide():
        return render_template('guide.html', mushrooms=MUSHROOM_SPECIES)

    @app.route('/api/mushroom/<mushroom_id>')
    def get_mushroom_info(mushroom_id):
        mushroom_data = MUSHROOM_SPECIES.get(mushroom_id)
        if not mushroom_data:
            return jsonify({"error": "Mushroom not found"}), 404
        
        try:
            description = get_wiki_description(mushroom_data['url'])
            image_path = os.path.join(app.config['WIKI_IMAGES'], f"{mushroom_id}.jpg")
            if os.path.exists(image_path):
                image_url = url_for('static', filename=f'wiki_images/{mushroom_id}.jpg')
            else:
                image_url = url_for('static', filename='test_images/what_is_grib.png')
            
            return jsonify({
                "id": mushroom_id,
                "name": mushroom_data['name'],
                "edible": mushroom_data['edible'],
                "description": description,
                "image": image_url,
                "wiki_url": mushroom_data['url']
            })
        except Exception as e:
            print(f"Error processing mushroom {mushroom_id}: {e}")
            return jsonify({
                "id": mushroom_id,
                "name": mushroom_data['name'],
                "edible": mushroom_data['edible'],
                "description": "Не удалось загрузить информацию",
                "image": url_for('static', filename='test_images/what_is_grib.png'),
                "wiki_url": mushroom_data['url']
            })

    @app.route('/predict', methods=['GET', 'POST'])
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
                
                predictions = classifier.predict_image(filepath)
                
                return render_template('predict.html',
                                    image_path=filepath,
                                    predictions=predictions)
        return render_template('predict.html')
    
    return app

app = create_app()
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
