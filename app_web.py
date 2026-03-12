
import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
import gdown
from tensorflow.keras.models import load_model

# -------------------------------
# Download model if not present
# -------------------------------
MODEL_PATH = "leaf_model.h5"

if not os.path.exists(MODEL_PATH):
    url = "https://drive.google.com/uc?id=1oOnRu6MPfHgpUa1aqetQUTeif2Ad3Up5"
    gdown.download(url, MODEL_PATH, quiet=False)

# -------------------------------
# Load trained model
# -------------------------------
model = tf.keras.models.load_model(
    MODEL_PATH,
    compile=False,
    custom_objects={}
)

# -------------------------------
# Class labels
# -------------------------------
classes = ["early_blight", "late_blight", "healthy"]

# -------------------------------
# Symptoms dictionary
# -------------------------------
symptoms = {
    "early_blight": "Brown spots with concentric rings on leaves.",
    "late_blight": "Dark irregular lesions that spread rapidly in humid conditions.",
    "healthy": "Leaf shows no disease symptoms."
}

# -------------------------------
# Treatment dictionary
# -------------------------------
treatment = {
    "early_blight": "Remove infected leaves and apply fungicide.",
    "late_blight": "Use copper fungicide and avoid overhead watering.",
    "healthy": "No treatment required."
}

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🌿 AI Plant Disease Detection System")
st.write("Upload or capture a leaf image to detect plant diseases.")

uploaded_file = st.file_uploader(
    "Upload Leaf Image",
    type=["jpg", "png", "jpeg", "webp"]
)

camera_image = st.camera_input("Or Take a Photo")

image = None

if uploaded_file is not None:
    image = Image.open(uploaded_file)

elif camera_image is not None:
    image = Image.open(camera_image)

# -------------------------------
# Prediction
# -------------------------------
if image is not None:

    st.image(image, caption="Selected Leaf Image", use_column_width=True)

    img = np.array(image)
    img = cv2.resize(img, (224, 224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    confidence = np.max(prediction) * 100
    result = classes[np.argmax(prediction)]

    st.success(f"Prediction: {result}")
    st.write(f"Confidence: {confidence:.2f}%")

    st.subheader("Symptoms")
    st.write(symptoms[result])

    st.subheader("Recommended Treatment")
    st.write(treatment[result])

    # -------------------------------
    # Probability Graph
    # -------------------------------
    probabilities = prediction[0]

    fig, ax = plt.subplots()
    ax.bar(classes, probabilities)
    ax.set_ylabel("Probability")
    ax.set_title("Prediction Confidence")

    st.pyplot(fig)















