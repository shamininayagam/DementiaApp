import os
import sys
import flask
from flask import request, jsonify
import numpy as np
import librosa
import joblib
import pickle
from werkzeug.utils import secure_filename

# Model loading function with robust error handling
def load_model(model_path):
    if not os.path.exists(model_path):
        print(f"Model file not found: {model_path}")
        return None
    
    try:
        # Try joblib first
        return joblib.load(model_path)
    except Exception as joblib_error:
        print(f"Joblib loading failed for {model_path}: {joblib_error}")
        
        try:
            # Fallback to pickle
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        except Exception as pickle_error:
            print(f"Pickle loading failed for {model_path}: {pickle_error}")
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
    print("/predict")
    print(request.files)
    # Explicit try-except block
    try:
        # Model validation
        if loaded_model is None or loaded_scaler is None:
            return jsonify({
                'error': 'Model not loaded. Check model files.',
                'status': 'error',
                'details': 'Model or scaler could not be loaded'
            }), 500

        # File handling
        if 'file' not in request.files:
            return jsonify({
                'error': 'No file uploaded',
                'status': 'error'
            }), 400

        file = request.files['file']
        if file.filename == '':
            return jsonify({
                'error': 'No selected file',
                'status': 'error'
            }), 400

        # Save file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Additional nested try-except for detailed error handling
        try:
            # Preprocessing
            y = preprocess_audio(filepath)
            if y is None:
                os.remove(filepath)
                return jsonify({
                    'error': 'Audio preprocessing failed',
                    'status': 'error'
                }), 400

            # Feature extraction
            features = extract_librosa_features(y)
            if features is None:
                os.remove(filepath)
                return jsonify({
                    'error': 'Feature extraction failed',
                    'status': 'error'
                }), 400

            # Prediction
            features = features.reshape(1, -1)
            features = loaded_scaler.transform(features)
            prediction = loaded_model.predict(features)
            proba = loaded_model.predict_proba(features)

            # Clean up file
            os.remove(filepath)

            # Map prediction
            class_names = {0: 'AD (Alzheimer\'s Disease)', 1: 'CN (Cognitively Normal)'}
            predicted_class = class_names.get(prediction[0], 'Unknown')

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
            # Ensure file is removed in case of processing error
            if os.path.exists(filepath):
                os.remove(filepath)
            
            # Log the full error
            print(f"Processing error: {process_error}")
            print(f"Error details: {sys.exc_info()}")
            
            return jsonify({
                'error': f'Audio processing error: {str(process_error)}',
                'status': 'error',
                'details': str(sys.exc_info())
            }), 500

    except Exception as main_error:
        # Catch any unexpected errors in the main route handler
        print(f"Unexpected error in prediction route: {main_error}")
        print(f"Error details: {sys.exc_info()}")
        
        return jsonify({
            'error': 'Unexpected error during prediction',
            'status': 'error',
            'details': str(main_error)
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
    # Print model loading status
    if loaded_model is None:
        print("WARNING: No model could be loaded. Check your model files.")
    if loaded_scaler is None:
        print("WARNING: No scaler could be loaded. Check your scaler files.")
    
    app.run(debug=True,port=5000)