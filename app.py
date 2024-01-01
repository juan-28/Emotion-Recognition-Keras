import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from PIL import Image
import numpy as np
from keras.preprocessing import image
from flask import Flask, request, jsonify, render_template, send_file
import seaborn as sns
import os
from werkzeug.utils import secure_filename
from image_utils import preprocess_image, apply_occlusion
from heatmap import generate_heatmap


app = Flask(__name__)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load your Keras model
model_path = os.path.join(BASE_DIR, 'pretrained_Resnet.h5')
model = load_model(model_path)

@app.route('/', methods=['GET'])
def index():
    # Render the upload form page
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})
    
    if file:
        # Define a path to save the image
        image_directory = '/tmp/images'
        if not os.path.exists(image_directory):
            os.makedirs(image_directory)
        image_path = os.path.join(image_directory, file.filename)
        file.save(image_path)

        # Process the image and get the prediction
        image = Image.open(image_path)
        processed_image = preprocess_image(image, target_size=(197, 197))
        prediction = model.predict(processed_image)
        predicted_class_index = np.argmax(prediction, axis=1)[0]
        class_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        predicted_class = class_labels[predicted_class_index]

        # Return the prediction and the path where the image was saved
        return jsonify({'prediction': predicted_class, 'image_path': image_path})

    return jsonify({'error': 'Error during prediction'})

@app.route('/heatmap', methods=['POST'])
def heatmap():
    if not request.is_json:
        print("No JSON received.")
        return jsonify({'error': 'Missing JSON in request'}), 400

    data = request.get_json()
    print("Data received:", data)  # Log the received data for debugging

    image_path = data.get('image_path')
    if not image_path:
        print("No image_path in JSON.")
        return jsonify({'error': 'Missing image_path'}), 400

    heatmap_path = generate_heatmap(image_path)
    if not os.path.exists(heatmap_path):
        print("Heatmap file not found at:", heatmap_path)
        return jsonify({'error': 'Heatmap generation failed'}), 500

    return send_file(heatmap_path, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
