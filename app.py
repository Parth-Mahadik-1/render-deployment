# app.py
import os
import io
import numpy as np
from flask import Flask, request, render_template, redirect, url_for, flash, jsonify
from werkzeug.utils import secure_filename
import tensorflow as tf
from PIL import Image

# ---------------------------
# Configuration
# ---------------------------
app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# ---------------------------
# Classes and model
# ---------------------------
CLASS_NAMES = [
    "Black Scurf",
    "Blackleg",
    "Common Scab",
    "Dry Rot",
    "Healthy",
    "Miscellaneous",
    "Pink Rot"
]

model = None

def load_trained_model():
    global model
    try:
        # Use the same model file you used with FastAPI
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(BASE_DIR, "model", "model_with_inference_2.h5")
        model = tf.keras.models.load_model(model_path)
        print("✅ Model loaded successfully.")
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        model = None

load_trained_model()
# ---------------------------
# Image read / prediction (FastAPI logic)
# ---------------------------
def read_file_as_image(data: bytes) -> np.ndarray:
    """
    EXACT same logic as your FastAPI read_file_as_image:
    - do NOT convert to RGB
    - do NOT resize/normalize
    - return raw numpy array from PIL Image
    """
    image = Image.open(io.BytesIO(data))
    return np.array(image)

def predict_from_bytes(image_bytes: bytes):
    """Run model prediction using raw image array (same as FastAPI)."""
    if model is None:
        return None, 0.0, None

    try:
        img = read_file_as_image(image_bytes)
        # batch dim exactly like your FastAPI code
        img_batch = np.expand_dims(img, 0)
        preds = model.predict(img_batch)
        idx = int(np.argmax(preds[0]))
        confidence = float(preds[0][idx])
        class_name = CLASS_NAMES[idx]
        return class_name, confidence, img.shape
    except Exception as e:
        print(f"Prediction error: {e}")
        return None, 0.0, None

# ---------------------------
# Helpers
# ---------------------------
def allowed_file(filename: str) -> bool:
    ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif', 'bmp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# ---------------------------
# Routes (2nd-app format)
# ---------------------------
@app.route('/')
def index():
    # if you have templates, this will render them; otherwise simple message
    try:
        return render_template('index.html')
    except Exception:
        return "Flask ML App (upload form expected). Use POST /predict or /api/predict."

@app.route('/predict', methods=['POST'])
def predict():
    """Form-based endpoint (HTML result page) — keeps the 2nd-app flow but uses FastAPI-style logic."""
    if 'file' not in request.files:
        flash('No file uploaded')
        return redirect(request.url)

    file = request.files['file']
    if file.filename == '':
        flash('No file selected')
        return redirect(request.url)

    if not allowed_file(file.filename):
        flash('Invalid file type')
        return redirect(url_for('index'))

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        with open(filepath, 'rb') as f:
            img_bytes = f.read()

        predicted_class, confidence, shape = predict_from_bytes(img_bytes)

    finally:
        # always clean up
        if os.path.exists(filepath):
            os.remove(filepath)

    if predicted_class is None:
        flash('Prediction failed. Check server logs.')
        return redirect(url_for('index'))

    result = {
        'predicted_class': predicted_class,
        'confidence': round(confidence * 100, 2),
        'filename': filename,
        'shape': list(shape) if shape is not None else None
    }

    # If you have result.html, this will render. Otherwise return JSON for convenience.
    try:
        return render_template('result.html', result=result)
    except Exception:
        return jsonify(result)



# ---------------------------
# Run
# ---------------------------
from flask_ngrok import run_with_ngrok
run_with_ngrok(app)  # automatically starts ngrok
app.run()
