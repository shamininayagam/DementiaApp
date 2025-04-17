import os
import sys
import flask
from flask import request, jsonify
import numpy as np
import librosa
import joblib
import pickle
from werkzeug.utils import secure_filename
import logging
import traceback
import socket

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def find_available_port(start_port=5000, max_port=5010):
    """Find an available port in the given range."""
    for port in range(start_port, max_port + 1):
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.bind(('127.0.0.1', port))
                return port
        except OSError:
            continue
    raise RuntimeError("No available ports found")

# Model loading function with robust error handling
def load_model(model_path):
    if not os.path.exists(model_path):
        logger.error(f"Model file not found: {model_path}")
        return None
    
    try:
        # Try joblib first
        return joblib.load(model_path)
    except Exception as joblib_error:
        logger.error(f"Joblib loading failed for {model_path}: {joblib_error}")
        
        try:
            # Fallback to pickle
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        except Exception as pickle_error:
            logger.error(f"Pickle loading failed for {model_path}: {pickle_error}")
            return None

# Attempt to load models
loaded_model = None
loaded_scaler = None

# List of possible model paths
model_paths = [
    'audio_classification_model.joblib',
    'audio_classification_model.pkl',
    # 'audio_model.joblib',
    # 'audio_model.pkl'
    'audio_scaler.joblib',
    'audio_scaler'
]

scaler_paths = [
    'audio_scaler.joblib',
    'audio_scaler.pkl',
    'scaler.joblib',
    'scaler.pkl'
]

# Try loading model
for path in model_paths:
    loaded_model = load_model(path)
    if loaded_model is not None:
        print(f"Loaded model from {path}")
        break

# Try loading scaler
for path in scaler_paths:
    loaded_scaler = load_model(path)
    if loaded_scaler is not None:
        print(f"Loaded scaler from {path}")
        break

# Create Flask app
app = flask.Flask(__name__)

# Configuration
UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'wav'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def preprocess_audio(file_path):
    try:
        y, sr = librosa.load(file_path, sr=22050)
        y = y.astype(np.float32)
        y = y / np.max(np.abs(y) + 1e-10)
        return y
    except Exception as e:
        print(f"Preprocessing error: {e}")
        return None

def extract_librosa_features(y):
    try:
        y = y.astype(np.float32)
        
        mfcc = librosa.feature.mfcc(y=y, sr=22050, n_mfcc=13)
        mfcc_mean = np.mean(mfcc, axis=1).astype(np.float32)
        mfcc_std = np.std(mfcc, axis=1).astype(np.float32)
        
        chroma = librosa.feature.chroma_stft(y=y, sr=22050)
        chroma_mean = np.mean(chroma, axis=1).astype(np.float32)
        chroma_std = np.std(chroma, axis=1).astype(np.float32)
        
        features = np.concatenate([
            mfcc_mean, mfcc_std, 
            chroma_mean, chroma_std
        ]).astype(np.float32)
        
        return features
    except Exception as e:
        print(f"Feature extraction error: {e}")
        return None

@app.route('/predict', methods=['POST'])
def predict_audio():
    logger.info("Received prediction request")
    logger.debug(f"Request files: {request.files}")
    logger.debug(f"Request headers: {request.headers}")
    
    try:
        if loaded_model is None or loaded_scaler is None:
            logger.error("Model or scaler not loaded")
            return jsonify({
                'error': 'Model not loaded. Check model files.',
                'status': 'error',
                'details': 'Model or scaler could not be loaded'
            }), 500

        if 'file' not in request.files:
            logger.error("No file in request")
            return jsonify({
                'error': 'No file uploaded',
                'status': 'error'
            }), 400

        file = request.files['file']
        if file.filename == '':
            logger.error("Empty filename")
            return jsonify({
                'error': 'No selected file',
                'status': 'error'
            }), 400

        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        logger.info(f"Saving file to: {filepath}")
        file.save(filepath)

        try:
            logger.info("Starting audio preprocessing")
            y = preprocess_audio(filepath)
            if y is None:
                logger.error("Audio preprocessing failed")
                os.remove(filepath)
                return jsonify({
                    'error': 'Audio preprocessing failed',
                    'status': 'error'
                }), 400

            logger.info("Starting feature extraction")
            features = extract_librosa_features(y)
            if features is None:
                logger.error("Feature extraction failed")
                os.remove(filepath)
                return jsonify({
                    'error': 'Feature extraction failed',
                    'status': 'error'
                }), 400

            logger.info("Starting prediction")
            features = features.reshape(1, -1)
            features = loaded_scaler.transform(features)
            prediction = loaded_model.predict(features)
            proba = loaded_model.predict_proba(features)

            # Clean up file
            os.remove(filepath)

            # Map prediction
            class_names = {0: 'AD (Alzheimer\'s Disease)', 1: 'CN (Cognitively Normal)'}
            predicted_class = class_names.get(prediction[0], 'Unknown')
            
            logger.info(f"Prediction successful: {predicted_class}")
            return jsonify({
                'prediction': int(prediction[0]),
                'class_name': predicted_class,
                'confidence': {
                    'AD': float(proba[0][0]),
                    'CN': float(proba[0][1])
                },
                'status': 'success'
            })

        except Exception as process_error:
            logger.error(f"Processing error: {process_error}")
            logger.error(traceback.format_exc())
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({
                'error': f'Audio processing error: {str(process_error)}',
                'status': 'error',
                'details': traceback.format_exc()
            }), 500

    except Exception as main_error:
        logger.error(f"Unexpected error in prediction route: {main_error}")
        logger.error(traceback.format_exc())
        return jsonify({
            'error': 'Unexpected error during prediction',
            'status': 'error',
            'details': traceback.format_exc()
        }), 500

@app.route('/', methods=['GET'])
def home():
    # Provide information about model loading
    model_status = 'Loaded' if loaded_model and loaded_scaler else 'Not Loaded'
    return jsonify({
        'message': 'Audio Classification API',
        'model_status': model_status,
        'endpoints': {
            '/predict': 'POST method to classify audio files'
        }
    })

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({
        'error': 'Endpoint not found',
        'status': 'error'
    }), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({
        'error': 'Internal server error',
        'status': 'error'
    }), 500

if __name__ == '__main__':
    try:
        if loaded_model is None:
            logger.warning("WARNING: No model could be loaded. Check your model files.")
        if loaded_scaler is None:
            logger.warning("WARNING: No scaler could be loaded. Check your scaler files.")
        
        port = 5000
        logger.info(f"Starting server on port {port}")
        
        with open('flask_port.txt', 'w') as f:
            f.write(str(port))
        
        app.run(debug=False, port=port, host='127.0.0.1', threaded=True)
    except Exception as e:
        logger.error(f"Failed to start server: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)