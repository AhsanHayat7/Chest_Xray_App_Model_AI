from flask import Flask, request, jsonify
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
from PIL import Image
import io
import os

app = Flask(__name__)

# Global variable for the model
model = None

def load_keras_model():
    """Load the fine-tuned model with proper error handling."""
    model_paths = [
        'best_model.h5',    # Best model during training
        'final_model.h5',   # Final model after fine-tuning
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            try:
                loaded_model = load_model(path)
                print(f"Model loaded successfully from {path}")
                return loaded_model
            except Exception as e:
                print(f"Error loading model from {path}: {e}")
    
    raise FileNotFoundError("No valid model file found in specified paths")

def preprocess_image(image):
    """Preprocess the image to match the model's requirements."""
    try:
        if image.mode != 'RGB':
            image = image.convert('RGB')
        image = image.resize((224, 224))  # Match IMAGE_SIZE from training
        image_array = np.array(image) / 255.0  # Normalize to [0, 1]
        return np.expand_dims(image_array, axis=0)
    except Exception as e:
        raise ValueError(f"Error preprocessing image: {e}")

@app.route('/', methods=['GET'])
def index():
    """Render the main page with improved UI."""
    return '''
    <!doctype html>
    <html>
    <head>
        <title>Chest X-ray Analysis</title>
        <style>
            body {
                font-family: Arial, sans-serif;
                max-width: 800px;
                margin: 0 auto;
                padding: 20px;
                background-color: #f5f5f5;
            }
            .container {
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 0 10px rgba(0,0,0,0.1);
            }
            h1 {
                color: #2c3e50;
                text-align: center;
                margin-bottom: 30px;
            }
            .upload-form {
                margin: 20px 0;
                padding: 20px;
                border: 2px dashed #3498db;
                border-radius: 10px;
                text-align: center;
            }
            .upload-form:hover {
                border-color: #2980b9;
            }
            input[type="file"] {
                margin: 10px 0;
            }
            input[type="submit"] {
                background-color: #3498db;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 5px;
                cursor: pointer;
                font-size: 16px;
            }
            input[type="submit"]:hover {
                background-color: #2980b9;
            }
            .result {
                margin-top: 20px;
                padding: 20px;
                border-radius: 5px;
                text-align: center;
            }
            .error {
                color: #e74c3c;
                background-color: #fde8e7;
                padding: 10px;
                border-radius: 5px;
                margin-top: 10px;
            }
            .success {
                color: #27ae60;
                background-color: #e8f8f5;
                padding: 10px;
                border-radius: 5px;
                margin-top: 10px;
            }
            #preview {
                max-width: 300px;
                margin: 20px auto;
                display: none;
            }
            .loading {
                display: none;
                text-align: center;
                margin: 20px 0;
            }
            .loading:after {
                content: "⚕️ Analyzing...";
                animation: dots 1.5s steps(5, end) infinite;
            }
            @keyframes dots {
                0%, 20% { content: "⚕️ Analyzing.  "; }
                40% { content: "⚕️ Analyzing.. "; }
                60% { content: "⚕️ Analyzing..."; }
                80% { content: "⚕️ Analyzing...."; }
                100% { content: "⚕️ Analyzing....."; }
            }
        </style>
        <script>
            function previewImage(input) {
                const preview = document.getElementById('preview');
                const file = input.files[0];
                const reader = new FileReader();

                reader.onload = function(e) {
                    preview.src = e.target.result;
                    preview.style.display = 'block';
                }

                if (file) {
                    reader.readAsDataURL(file);
                }
            }
            
            function showLoading() {
                document.querySelector('.loading').style.display = 'block';
            }
            
            function hideLoading() {
                document.querySelector('.loading').style.display = 'none';
            }

            async function uploadImage(event) {
                event.preventDefault();
                const formData = new FormData(document.getElementById('upload-form'));
                const resultDiv = document.getElementById('result');
                
                showLoading();
                resultDiv.innerHTML = '';
                
                try {
                    const response = await fetch('/', {
                        method: 'POST',
                        body: formData
                    });
                    const data = await response.json();
                    
                    hideLoading();
                    
                    if (data.error) {
                        resultDiv.innerHTML = `<div class="error">Error: ${data.error}</div>`;
                    } else {
                        const resultClass = data.prediction === 'Infected' ? 'error' : 'success';
                        resultDiv.innerHTML = `
                            <div class="${resultClass}">
                                <h3>Analysis Result</h3>
                                <p>Prediction: ${data.prediction}</p>
                                <p>Confidence: ${data.confidence}</p>
                            </div>
                        `;
                    }
                } catch (error) {
                    hideLoading();
                    resultDiv.innerHTML = `<div class="error">Error: ${error.message}</div>`;
                }
            }
        </script>
    </head>
    <body>
        <div class="container">
            <h1>🫁 Chest X-ray Analysis</h1>
            <div class="upload-form">
                <form id="upload-form" onsubmit="uploadImage(event)" enctype="multipart/form-data">
                    <input type="file" name="file" accept="image/*" required onchange="previewImage(this)">
                    <br>
                    <input type="submit" value="Analyze Image">
                </form>
                <img id="preview" alt="Image preview">
            </div>
            <div class="loading"></div>
            <div id="result" class="result"></div>
        </div>
    </body>
    </html>
    '''

@app.route('/', methods=['POST'])
def analyze_image():
    """Handle image upload and analysis."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    try:
        # Read and preprocess the image
        image = Image.open(io.BytesIO(file.read()))
        processed_image = preprocess_image(image)
        
        # Make prediction
        prediction = model.predict(processed_image, verbose=0)
        confidence = float(prediction[0][0])
        
        # Create result with confidence score
        result = {
            'prediction': 'Infected' if confidence > 0.5 else 'Not Infected',
            'confidence': f"{abs(confidence - 0.5) * 200:.2f}%"  # Adjusted confidence calculation
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    try:
        # Load the model before starting the app
        model = load_keras_model()
        # Start the Flask app
        app.run(debug=True)
    except Exception as e:
        print(f"Fatal error: Could not load model: {e}")