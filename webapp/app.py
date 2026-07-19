import os
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_file, render_template
from flask_cors import CORS
from flask_socketio import SocketIO, emit
from flask_sqlalchemy import SQLAlchemy
import tensorflow as tf
from werkzeug.utils import secure_filename
import uuid
import threading
import time
from PIL import Image
from datetime import datetime
import json

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key'
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///tasks.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*")
db = SQLAlchemy(app)

UPLOAD_FOLDER = 'uploads'
STATIC_FOLDER = 'static'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'bmp', 'tiff', 'tif', 'webp', 'jfif'}

# Путь к модели — теперь относительный
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'models', 'enhancer_model.keras')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# Модель БД
class Task(db.Model):
    id = db.Column(db.String(64), primary_key=True)
    filename = db.Column(db.String(256))
    status = db.Column(db.String(32), default='pending')
    progress = db.Column(db.Integer, default=0)
    result_url = db.Column(db.String(256), nullable=True)
    coeffs_json = db.Column(db.Text, nullable=True)
    error = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'status': self.status,
            'progress': self.progress,
            'result_url': self.result_url,
            'coefficients': json.loads(self.coeffs_json) if self.coeffs_json else None,
            'error': self.error,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }

with app.app_context():
    db.create_all()

# Загружаем модель
print("Loading model...")
print(f"Model path: {MODEL_PATH}")
if os.path.exists(MODEL_PATH):
    print("Model file found.")
else:
    print("ERROR: Model file NOT found!")
model = tf.keras.models.load_model(MODEL_PATH)
print("Model loaded")

# Очередь задач
task_queue = []
queue_lock = threading.Lock()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def read_image(filepath):
    try:
        img = cv2.imread(filepath)
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except:
        pass
    try:
        pil_img = Image.open(filepath)
        return np.array(pil_img.convert('RGB'))
    except:
        return None

def enhance_image(original_img):
    small = cv2.resize(original_img, (128, 128)) / 255.0
    coeffs = model.predict(small.reshape(1, 128, 128, 3), verbose=0)[0]
    
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

def process_task(task_id):
    with app.app_context():
        print(f"Processing task {task_id}")
        task = Task.query.get(task_id)
        if not task:
            print(f"Task {task_id} not found in DB")
            return
        
        task.status = 'processing'
        task.progress = 0
        db.session.commit()
        socketio.emit('task_update', task.to_dict(), room=task_id)
        
        try:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], task.filename)
            print(f"Reading file: {filepath}")
            original = read_image(filepath)
            if original is None:
                raise Exception("Could not read image")
            
            task.progress = 30
            db.session.commit()
            socketio.emit('task_update', task.to_dict(), room=task_id)
            print(f"Task {task_id}: image read, progress 30%")
            
            enhanced, coeffs = enhance_image(original)
            print(f"Task {task_id}: image enhanced")
            
            task.progress = 80
            db.session.commit()
            socketio.emit('task_update', task.to_dict(), room=task_id)
            
            result_filename = f"enhanced_{uuid.uuid4().hex}.jpg"
            result_path = os.path.join(STATIC_FOLDER, result_filename)
            cv2.imwrite(result_path, cv2.cvtColor(enhanced, cv2.COLOR_RGB2BGR))
            print(f"Task {task_id}: result saved to {result_path}")
            
            os.remove(filepath)
            
            task.status = 'done'
            task.progress = 100
            task.result_url = f'/static/{result_filename}'
            task.coeffs_json = json.dumps({
                'brightness': round(float(coeffs[0]), 3),
                'contrast': round(float(coeffs[1]), 3),
                'saturation': round(float(coeffs[2]), 3)
            })
            db.session.commit()
            socketio.emit('task_update', task.to_dict(), room=task_id)
            print(f"Task {task_id}: done")
            
        except Exception as e:
            print(f"Task {task_id} error: {str(e)}")
            task.status = 'error'
            task.error = str(e)
            db.session.commit()
            socketio.emit('task_update', task.to_dict(), room=task_id)
        finally:
            with queue_lock:
                if task_id in task_queue:
                    task_queue.remove(task_id)

def queue_worker():
    print("Queue worker started")
    while True:
        task_id = None
        with queue_lock:
            if task_queue:
                task_id = task_queue.pop(0)
                print(f"Queue worker: picked task {task_id}, queue size: {len(task_queue)}")
        
        if task_id:
            process_task(task_id)
        else:
            time.sleep(0.5)

# Запускаем воркер
time.sleep(1)  # Даём время на инициализацию
worker_thread = threading.Thread(target=queue_worker, daemon=True)
worker_thread.start()
print("Worker thread started")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/task', methods=['POST'])
def create_task():
    if 'file' not in request.files:
        return jsonify({'error': 'No file'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    if not allowed_file(file.filename):
        return jsonify({'error': 'Format not supported'}), 400
    
    task_id = uuid.uuid4().hex
    filename = secure_filename(file.filename)
    unique_name = f"{task_id}_{filename}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], unique_name)
    file.save(filepath)
    print(f"Task {task_id}: file saved to {filepath}")
    
    task = Task(
        id=task_id,
        filename=unique_name,
        status='pending',
        progress=0
    )
    db.session.add(task)
    db.session.commit()
    print(f"Task {task_id}: created in DB")
    
    with queue_lock:
        task_queue.append(task_id)
        print(f"Task {task_id}: added to queue, queue size: {len(task_queue)}")
    
    return jsonify({'task_id': task_id})

@app.route('/task/<task_id>/status', methods=['GET'])
def get_status(task_id):
    task = Task.query.get(task_id)
    if not task:
        return jsonify({'error': 'Task not found'}), 404
    return jsonify(task.to_dict())

@app.route('/task/<task_id>', methods=['DELETE'])
def abort_task(task_id):
    task = Task.query.get(task_id)
    if not task:
        return jsonify({'error': 'Task not found'}), 404
    if task.status in ['done', 'error']:
        return jsonify({'error': 'Task already finished'}), 400
    
    with queue_lock:
        if task_id in task_queue:
            task_queue.remove(task_id)
    
    task.status = 'aborted'
    db.session.commit()
    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], task.filename)
    if os.path.exists(filepath):
        os.remove(filepath)
    
    socketio.emit('task_update', task.to_dict(), room=task_id)
    return jsonify({'success': True})

@app.route('/task/<task_id>/result', methods=['GET'])
def get_result(task_id):
    task = Task.query.get(task_id)
    if not task:
        return jsonify({'error': 'Task not found'}), 404
    if task.status != 'done':
        return jsonify({'error': 'Task not finished'}), 400
    if not task.result_url:
        return jsonify({'error': 'Result not available'}), 404
    
    filename = os.path.basename(task.result_url)
    return send_file(os.path.join(STATIC_FOLDER, filename))

@app.route('/static/<filename>')
def get_static(filename):
    return send_file(os.path.join(STATIC_FOLDER, filename))

@socketio.on('connect')
def handle_connect():
    print('Client connected')

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

@socketio.on('subscribe')
def handle_subscribe(data):
    task_id = data.get('task_id')
    if task_id:
        task = Task.query.get(task_id)
        if task:
            emit('task_update', task.to_dict())

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, debug=True, host='0.0.0.0', port=port, allow_unsafe_werkzeug=True)