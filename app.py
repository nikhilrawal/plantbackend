import os
import numpy as np
from flask import Flask, request, jsonify
from PIL import Image
from tensorflow.keras.models import load_model

# Step 1: Disable GPU to avoid CUDA-related errors and enforce CPU usage
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

# Initialize Flask app
app = Flask(__name__)

# Load the .h5 model
model = load_model('plant_disease_model.h5')

# Class labels (add all the classes you have in the dataset)
class_labels = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
    'Blueberry___healthy', 'Cherry___Powdery_mildew', 'Cherry___healthy',
    'Corn___Cercospora_leaf_spot', 'Corn___Common_rust', 'Corn___Northern_Leaf_Blight',
    'Corn___healthy', 'Grape___Black_rot', 'Grape___Esca', 'Grape___Leaf_blight',
    'Grape___healthy', 'Orange___Citrus_greening', 'Peach___Bacterial_spot', 'Peach___healthy',
    'Pepper___Bacterial_spot', 'Pepper___healthy', 'Potato___Early_blight', 'Potato___Late_blight',
    'Potato___healthy', 'Strawberry___Leaf_scorch', 'Strawberry___healthy',
    'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight',
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites',
    'Tomato___Target_Spot', 'Tomato___Yellow_Leaf_Curl_Virus', 'Tomato___mosaic_virus',
    'Tomato___healthy'
]

# Preprocessing function
def preprocess_image(image, target_size):
    """
    Preprocess the image to the required input size for the model.
    """
    image = image.resize(target_size)  # Resize to the model's input size
    image_array = np.array(image) / 255.0  # Normalize pixel values
    image_array = np.expand_dims(image_array, axis=0).astype(np.float32)  # Add batch dimension
    return image_array

# Prediction endpoint
@app.route('/predict', methods=['POST'])
def predict():
    file = request.files.get('file')  # Get the uploaded file
    if not file:
        return jsonify({'error': 'No file uploaded'}), 400

    try:
        # Open and preprocess the image
        image = Image.open(file)
        preprocessed_image = preprocess_image(image, target_size=(128, 128))

        # Run inference with the .h5 model
        predictions = model.predict(preprocessed_image)

        # Get the predicted class and confidence
        predicted_class_index = np.argmax(predictions)
        predicted_class = class_labels[predicted_class_index]
        confidence = float(np.max(predictions))

        # Return the result
        return jsonify({'class': predicted_class, 'confidence': confidence})

    except Exception as e:
        return jsonify({'error': str(e)}), 500

# Main entry point
if __name__ == '__main__':
    # Render requires the app to bind to the PORT environment variable
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)

