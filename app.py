import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Load model
model = tf.keras.models.load_model("bone_model.h5")

IMG_SIZE = 160

st.title("🦴 Bone Abnormality Detection (Osteoporosis Risk Screening)")

uploaded_file = st.file_uploader("Upload X-ray image", type=["jpg","jpeg","png"])

if uploaded_file:
    img = Image.open(uploaded_file).resize((IMG_SIZE, IMG_SIZE))
    st.image(img, caption="Uploaded Image")

    img_array = np.array(img)/255.0
    img_array = np.expand_dims(img_array, axis=0)

    pred = model.predict(img_array)[0][0]

    if pred > 0.5:
        st.error("⚠️ Abnormal (Possible Fracture / Risk)")
    else:
        st.success("✅ Normal")

st.markdown("⚠️ This is a screening tool, not a medical diagnosis.")