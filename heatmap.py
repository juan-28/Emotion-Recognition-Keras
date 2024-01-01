import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from image_utils import preprocess_image, apply_occlusion
import numpy as np
import os
import seaborn as sns
from werkzeug.utils import secure_filename
from google.cloud import storage



BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Load your Keras model
model_path = os.path.join(BASE_DIR, 'pretrained_Resnet.h5')
model = load_model(model_path)

def generate_heatmap(image_path):
    original_image = load_img(image_path, target_size=(197, 197))
    original_image = img_to_array(original_image)
    original_image /= 255

    occluded_images, occlusion_locations = apply_occlusion(original_image)
    predictions = model.predict(occluded_images)
    original_prediction = model.predict(np.array([original_image]))

    heatmap = np.zeros(original_image.shape[:2])
    for idx, (prediction, location) in enumerate(zip(predictions, occlusion_locations)):
        change = np.sum(np.abs(original_prediction - prediction))
        heatmap[location[0]:location[0]+10, location[1]:location[1]+10] = change
    
    plt.figure(figsize=(8, 8))
    sns.heatmap(heatmap, cmap='viridis')
    plt.axis('off')

    # Save heatmap to a temporary file
    heatmap_path = 'tmp/images/heatmap.png'
    plt.savefig(heatmap_path, bbox_inches='tight', pad_inches=0)
    plt.close()

    return heatmap_path