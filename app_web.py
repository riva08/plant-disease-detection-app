import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
import os
import gdown


if not os.path.exists("leaf_model.h5"):
    gdown.download("https://drive.google.com/uc?id=1oOnRu6MPfHgpUa1aqetQUTeif2Ad3Up5", "leaf_model.h5", quiet=False)


tf.keras.backend.clear_session()

model = load_model("leaf_model.h5", compile=False)

symptoms = {
"early_blight": "Brown spots with concentric rings on leaves.",
"late_blight": "Dark irregular lesions that spread rapidly in humid conditions.",
"healthy": "Leaf shows no disease symptoms."
}

treatment = {
"early_blight": "Remove infected leaves and apply fungicide.",
"late_blight": "Use copper fungicide and avoid overhead watering.",
"healthy": "No treatment required."
}

st.title("🌿 AI Plant Disease Detection System")
st.write("Upload or capture a leaf image to detect plant diseases.")

uploaded_file = st.file_uploader("Upload Leaf Image", type=["jpg","png","jpeg","webp"])

camera_image = st.camera_input("Or Take a Photo")

image = None

if uploaded_file is not None:
    image = Image.open(uploaded_file)

elif camera_image is not None:
    image = Image.open(camera_image)
if image is not None:

    st.image(image, caption="Selected Leaf Image", use_column_width=True)

    img = np.array(image)
    img = cv2.resize(img,(224,224))
    img = img/255.0
    img = np.expand_dims(img,axis=0)

    prediction = model.predict(img)

    confidence = np.max(prediction)*100
    result = classes[np.argmax(prediction)]

    st.success(f"Prediction: {result}")
    st.write(f"Confidence: {confidence:.2f}%")

    st.subheader("Symptoms")
    st.write(symptoms[result])

    st.subheader("Recommended Treatment")
    st.write(treatment[result])
    probabilities = prediction[0]

    fig, ax = plt.subplots()
    ax.bar(classes, probabilities)
    ax.set_ylabel("Probability")
    ax.set_title("Prediction Confidence")

    st.pyplot(fig)

    





