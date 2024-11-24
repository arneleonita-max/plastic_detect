from flask import Flask, request, jsonify
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import numpy as np
from PIL import Image
import os
from werkzeug.utils import secure_filename

app = Flask(__name__)

model = load_model('my_model.h5')

class_info = {
    0: {"name": "HDPE (High-Density Polyethylene)", "description": "A widely used plastic, often used for bottles, milk jugs, and detergent containers."},
    1: {"name": "PET (Polyethylene Terephthalate)", "description": "Commonly used for beverage bottles and food containers."},
    2: {"name": "PP (Polypropylene)", "description": "Used for food containers, straws, and bottle caps."},
    3: {"name": "PS (Polystyrene)", "description": "Used for disposable cups, plates, and plastic food containers."}
}

ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'png'}
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def preprocess_image(image, target_size=(224, 224)):
    image = image.resize(target_size)
    image = img_to_array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)
    return image

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file format. Supported formats: jpg, jpeg, png'}), 400
    
    try:
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)
        
        image = Image.open(file)
        processed_image = preprocess_image(image)
        
        predictions = model.predict(processed_image)
        
        predicted_class = np.argmax(predictions, axis=1)[0]
        probability = predictions[0][predicted_class]
        
        predicted_class = int(predicted_class)
        probability = float(probability)
        
        class_details = class_info.get(predicted_class, {"name": "Unknown", "description": "No information available"})
        
        response = {
            "predicted_class": predicted_class,
            "class_name": class_details["name"],
            "class_description": class_details["description"],
            "probability": probability
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(debug=True, host="127.0.0.1", port=8080)
