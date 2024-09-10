import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import img_to_array
from PIL import Image

st.header('Grand Challenge 7 Model To Predict Road Deployment')
st.write("""
Created by Yuzal

Model to Predict Plain or Pothole.
""")

# Load Model
path = "model.keras"
model_best = tf.keras.models.load_model(path)

gambar = st.file_uploader("Upload Gambar", type=['png', 'jpg', 'jpeg'])

# Func.Predict
def predict(gambar):
    # Buka gambar menggunakan PIL dan ubah ukurannya
    img = Image.open(gambar)
    img = img.resize((150, 150))
    
    # Ubah gambar menjadi array numpy
    img_array = img_to_array(img)
    
    # Tambahkan dimensi batch dan normalisasi gambar
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0

    # Lakukan prediksi
    prediction = model_best.predict(img_array)
    return prediction

if gambar is not None:
    prediksi = predict(gambar)
    if prediksi > 0.5:
        ans = "Pothole"
    else:
        ans = "Plain"

    st.write('Based on user input, the model predicted:')
    st.write(ans)
