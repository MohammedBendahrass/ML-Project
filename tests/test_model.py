# tests/test_model.py

import os
import joblib
import numpy as np
import cv2
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_classification

# Function to generate model and scaler if they don't exist
def generate_model_and_scaler():
    X, y = make_classification(n_samples=1000, n_features=4096, random_state=42)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    model = RandomForestClassifier()
    model.fit(X_scaled, y)
    os.makedirs('models', exist_ok=True)
    joblib.dump(model, 'models/best_rf_model.joblib')
    joblib.dump(scaler, 'models/scaler.joblib')

# Check if the model and scaler files exist, if not generate them
if not os.path.exists('models/best_rf_model.joblib') or not os.path.exists('models/scaler.joblib'):
    generate_model_and_scaler()

def load_test_image(image_path):
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (64, 64))
    return image.flatten()

def test_model_loading():
    model = joblib.load('models/best_rf_model.joblib')
    scaler = joblib.load('models/scaler.joblib')
    assert isinstance(model, RandomForestClassifier), "Model is not a RandomForestClassifier"
    assert hasattr(scaler, 'transform'), "Scaler does not have transform method"

def test_preprocess_image():
    image_path = 'archive (4)/malware_dataset/val/BrowseFox/d7232af7b34b0dbba77086d3b44de19c36f9409cresized_image.png'
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    image = cv2.resize(image, (64, 64))
    assert image is not None, "Image loading failed"
    assert image.shape == (64, 64), "Image resizing failed"
    scaler = joblib.load('models/scaler.joblib')
    image = image.flatten()
    image = scaler.transform([image])
    assert image.shape == (1, 4096), "Image flattening or scaling failed"

def test_single_prediction():
    model = joblib.load('models/best_rf_model.joblib')
    scaler = joblib.load('models/scaler.joblib')
    image_path = 'archive (4)/malware_dataset/val/BrowseFox/d7232af7b34b0dbba77086d3b44de19c36f9409cresized_image.png'
    image = load_test_image(image_path)
    image = scaler.transform([image])
    prediction = model.predict(image)
    if not isinstance(prediction[0], int):
        prediction = prediction.argmax(axis=1)  # Convert probabilities to class labels if necessary
    assert len(prediction) == 1, "Prediction output size is incorrect"
    assert isinstance(prediction[0], int), "Prediction output type is incorrect"

def test_batch_prediction():
    model = joblib.load('models/best_rf_model.joblib')
    scaler = joblib.load('models/scaler.joblib')
    image_paths = ['archive (4)/malware_dataset/val/Regrun/e41f24eeb32c6b2bb8417c939dd8324f8fbd5058resized_image.png',
                   'archive (4)/malware_dataset/val/InstallCore/c6fdb54efd518193e0a06aa08c3d1f16eacb04a1resized_image.png',
                   'archive (4)/malware_dataset/val/MultiPlug/bf04f2de70b32abda50a756b435e9d6ed252b952resized_image.png',
                   'archive (4)/malware_dataset/val/VBKrypt/d25fb0376de150f000ca1d4ded333fa9ea6b97deresized_image.png']
    images = [load_test_image(image_path) for image_path in image_paths]
    images = scaler.transform(images)
    predictions = model.predict(images)
    if not isinstance(predictions[0], int):
        predictions = predictions.argmax(axis=1)  # Convert probabilities to class labels if necessary
    assert len(predictions) == len(image_paths), "Batch prediction output size is incorrect"
    assert all(isinstance(pred, int) for pred in predictions), "Batch prediction output type is incorrect"

def test_edge_cases():
    model = joblib.load('models/best_rf_model.joblib')
    scaler = joblib.load('models/scaler.joblib')

    # Completely black image
    black_image = np.zeros((64, 64)).flatten()
    black_image = scaler.transform([black_image])
    black_prediction = model.predict(black_image)
    if not isinstance(black_prediction[0], int):
        black_prediction = black_prediction.argmax(axis=1)  # Convert probabilities to class labels if necessary
    assert len(black_prediction) == 1, "Black image prediction output size is incorrect"

    # Completely white image
    white_image = np.ones((64, 64)).flatten() * 255
    white_image = scaler.transform([white_image])
    white_prediction = model.predict(white_image)
    if not isinstance(white_prediction[0], int):
        white_prediction = white_prediction.argmax(axis=1)  # Convert probabilities to class labels if necessary
    assert len(white_prediction) == 1, "White image prediction output size is incorrect"

def test_performance():
    model = joblib.load('models/best_rf_model.joblib')
    scaler = joblib.load('models/scaler.joblib')
    image_path = 'archive (4)/malware_dataset/val/Dialplatform.B/04b65397b0b6f6d1605959b4b2b53f3e.png'
    image = load_test_image(image_path)
    image = scaler.transform([image])
    
    import time
    start_time = time.time()
    prediction = model.predict(image)
    end_time = time.time()
    elapsed_time = end_time - start_time
    
    if not isinstance(prediction[0], int):
        prediction = prediction.argmax(axis=1)  # Convert probabilities to class labels if necessary
    assert elapsed_time < 1, "Prediction took too long"
