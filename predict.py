import tensorflow as tf
import numpy as np
import cv2

model = tf.keras.models.load_model("leaf_model.h5")

classes = ["early_blight", "late_blight", "healthy"]

img = cv2.imread("test.jpg")
img = cv2.resize(img, (224,224))
img = img / 255.0
img = np.expand_dims(img, axis=0)

prediction = model.predict(img)

confidence = np.max(prediction)
result = classes[np.argmax(prediction)]
print("Prediction:", result)
print("Confidence:", confidence)

print("Predicted disease:", result)