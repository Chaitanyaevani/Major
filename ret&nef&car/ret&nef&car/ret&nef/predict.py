import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the pre-trained model
model_path = r'C:\Users\ANUHYA ADUSUMILLI\Desktop\mini-(2,12)\my_model(260624-1650).h5'
model = load_model(model_path, compile=False)

# Define class labels based on your dataset
class_names = ['NO_DR', 'Mild', 'Moderate', 'Proliferate', 'Severe']

# Function to preprocess the image
def preprocess_image(image_path, target_size=(128, 128)):
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, target_size)
    image = image.astype('float32') / 255.0  # Normalize pixel values
    return image

# Function to predict class labels for a folder
def predict_folder(folder_path):
    class_labels = []
    for image_file in os.listdir(folder_path):
        if image_file.endswith('.png'):  # Assuming images are in PNG format
            image_path = os.path.join(folder_path, image_file)
            image = preprocess_image(image_path)
            image = np.expand_dims(image, axis=0)  # Add batch dimension
            prediction = model.predict(image)
            class_label = np.argmax(prediction, axis=1)[0]
            class_labels.append(class_names[class_label])  # Map numerical label to class name
    return class_labels

# Single folder containing images to predict
folder_path = r'C:\Users\ANUHYA ADUSUMILLI\Desktop\mini-(2,12)\dataset\gaussian_filtered_images\Proliferate'

# Predict class labels for the folder
class_labels = predict_folder(folder_path)
print(f'Class labels for folder {folder_path}: {class_labels}')
