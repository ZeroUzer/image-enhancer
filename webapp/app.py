import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from werkzeug.utils import secure_filename
import uuid
import time
from PIL import Image
import gc
import sys
from tflite_runtime.interpreter import Interpreter

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///tasks.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

CORS(app)

UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif', 'webp', 'jfif'}

MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'enhancer_model.tflite')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Загрузка модели (без tf)
print("Loading TFLite model...")
try:
    interpreter = Interpreter(model_path=MODEL_PATH)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("TFLite model loaded")
except Exception as e:
    print(f"ERROR loading model: {e}")
    sys.exit(1)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def compress_image(filepath, max_pixels=5_000_000):
    try:
        img = Image.open(filepath)
        w, h = img.size
        if w * h > max_pixels:
            scale = (max_pixels / (w * h)) ** 0.5
            new_w, new_h = int(w * scale), int(h * scale)
            img = img.resize((new_w, new_h), Image.LANCZOS)
            img.save(filepath, quality=85, optimize=True)
            print(f"Image compressed from {w}x{h} to {new_w}x{new_h}")
        else:
            img.save(filepath, quality=85, optimize=True)
    except Exception as e:
        print(f"Compression error: {e}")

def read_image(filepath):
    try:
        img = cv2.imread(filepath)
        if img is None:
            return None
        h, w = img.shape[:2]
        if w * h > 8_000_000:
            scale = (8_000_000 / (w * h)) ** 0.5
            new_w, new_h = int(w * scale), int(h * scale)
            img = cv2.resize(img, (new_w, new_h))
            print(f"Image resized from {w}x{h} to {new_w}x{new_h}")
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Read error: {e}")
        return None

def enhance_image(original_img):
    try:
        small = cv2.resize(original_img, (128, 128)) / 255.0
        input_data = np.expand_dims(small, axis=0).astype(np.float32)
        
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        coeffs = interpreter.get_tensor(output_details[0]['index'])[0]
        
        k_brightness = 0.5 + coeffs[0] * 1.5
        k_contrast = 0.5 + coeffs[1] * 1.5
        k_saturation = coeffs[2] * 2.0
        
        result = original_img.astype(np.float32) / 255.0
        result = result * k_contrast + (128/255.0) * (1 - k_contrast) + (k_brightness - 1) * 0.5
        result = np.clip(result, 0, 1)
        
        hsv = cv2.cvtColor((result * 255).astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        hsv[:, :, 1] = hsv[:, :, 1] * k_saturation
        hsv[:, :, 1] = np.clip(hsv[:, :, 1], 0, 255)
        result = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
        
        return result, coeffs
    except Exception as e:
        print(f"Enhance error: {e}")
        raise

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    start_time = time.time()
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Format not supported'}), 400
    
    try:
        filename = secure_filename(file.filename)
        unique_name = f"{uuid.uuid4().hex}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
        
        file.save(filepath)
        print(f"File saved: {filepath}")
        
        compress_image(filepath)
        
        original = read_image(filepath)
        if original is None:
            os.remove(filepath)
            return jsonify({'error': 'Could not read image'}), 400
        
        print(f"Image shape: {original.shape}")
        
        enhanced, coeffs = enhance_image(original)
        print("Enhance complete")
        
        result_filename = f"enhanced_{uuid.uuid4().hex}.jpg"
        result_path = os.path.join(STATIC_FOLDER, result_filename)
        cv2.imwrite(result_path, cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR))
        
        os.remove(filepath)
        
        del original
        del enhanced
        gc.collect()
        
        elapsed = time.time() - start_time
        print(f"Total time: {elapsed:.2f}s")
        
        return jsonify({
            'success': True,
            'result_url': f'/static/{result_filename}',
            'coefficients': {
                'brightness': round(float(coeffs[0]), 3),
                'contrast': round(float(coeffs[1]), 3),
                'saturation': round(float(coeffs[2]), 3)
            },
            'time': round(elapsed, 2)
        })
        
    except Exception as e:
        print(f"Upload error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/static/<filename>')
def get_static(filename):
    return send_file(os.path.join(STATIC_FOLDER, filename))

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    print(f"Starting on port {port}")
    app.run(debug=False, host='0.0.0.0', port=port)