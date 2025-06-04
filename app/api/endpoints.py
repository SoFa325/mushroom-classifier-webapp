from flask import Blueprint, request, jsonify
from app.api.models import predict_mushroom
import os

api_blueprint = Blueprint('api', __name__)

@api_blueprint.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename"}), 400
    
    # Сохраняем временный файл
    upload_dir = os.path.join(os.path.dirname(__file__), '../../static/uploads')
    os.makedirs(upload_dir, exist_ok=True)
    filepath = os.path.join(upload_dir, file.filename)
    file.save(filepath)
    
    # Получаем предсказание
    result = predict_mushroom(filepath)
    
    return jsonify(result)