import os
import json
from PIL import Image
import numpy as np
import tensorflow as tf
import streamlit as st
import gdown

# Function to download the model from Google Drive
def download_model_from_drive():
    file_id = '1wOMauFD_dPkFQAcvqjU9W-I2ELzfgiZq'  # Replace with your Google Drive file ID
    model_path = f"{working_dir}/trained_model/plant_disease_model.h5"
    if not os.path.exists(model_path):
        url = f'https://drive.google.com/uc?id={file_id}'
        gdown.download(url, model_path, quiet=False)
    return model_path

# Working directory path
working_dir = os.path.dirname(os.path.abspath(__file__))

# Download the model if not already available
model_path = download_model_from_drive()

# Load the PreTrained Model
model = tf.keras.models.load_model(model_path)

# Load the class indices
class_indices = json.load(open(f"{working_dir}/class_indices.json"))

# Function to Load and Preprocess the Image using PIL
def load_and_preprocess_image(image_path, target_size=(224,224)):
    # Load the Image
    img = Image.open(image_path)
    # Resize the Image
    img = img.resize(target_size)
    # Convert the image to numpy array
    img_array = np.array(img)
    # Add Batch Dimension
    img_array = np.expand_dims(img_array, axis=0)
    # Scale the image values to [0,1]
    img_array = img_array.astype('float32') / 255.0
    return img_array

# Function to Predict the Class of an Image
def predict_image_class(model, image_path, class_indices):
    preprocessed_image = load_and_preprocess_image(image_path)
    predictions = model.predict(preprocessed_image)
    predicted_class_index = np.argmax(predictions, axis=1)[0]
    predicted_class_name = class_indices[str(predicted_class_index)]
    return predicted_class_name

# Streamlit UI
st.title('🌿Plant Disease Classifier')

uploaded_image = st.file_uploader("Upload an Image...", type=['jpg', 'jpeg', 'png'])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    col1, col2 = st.columns(2)

    with col1:
        resized_img = image.resize((250, 250))
        st.image(resized_img)

    with col2:
        if st.button('Classify'):
            # Preprocess the Uploaded Image and Predict the Class
            prediction = predict_image_class(model, uploaded_image, class_indices)
            st.success(f"Prediction: {str(prediction)}")

st.text("Developed by Hariprasad")
