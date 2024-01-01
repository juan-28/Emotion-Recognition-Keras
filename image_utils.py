import numpy as np
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image


def preprocess_image(image, target_size):
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image /= 255
    return image

def apply_occlusion(image, size=10):
    occluded_images = []
    occlusion_locations = []
    
    for row in range(0, image.shape[0], size):
        for col in range(0, image.shape[1], size):
            occluded_image = image.copy()
            occluded_image[row:row+size, col:col+size] = 0
            occluded_images.append(occluded_image)
            occlusion_locations.append((row, col))
    
    return np.array(occluded_images), occlusion_locations

