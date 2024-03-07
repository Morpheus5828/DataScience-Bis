import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import numpy as np
from PIL import Image
import cv2

st.title("SmartBin")


@st.cache_data()
def load():
    model_path = "project/best_model.h5"
    model = load_model(model_path, compile=False)
    return model


model = load()


def predict(upload):
    img = Image.open(upload)
    img = img.convert('RGB')
    img = np.asarray(img)
    img_resize = cv2.resize(img, (224, 224))
    img_resize = np.expand_dims(img_resize, axis=0)
    pred = model.predict(img_resize)
    rec = pred[0][0]
    return rec


upload = st.file_uploader("Upload img", type=['png', 'jpeg', 'jpg'])

c1, c2 = st.columns(2)

if upload:
    rec = predict(upload)
    prob_recyclabe = rec * 100
    prob_organic = (1 - rec) * 100

    c1.image(Image.open(upload))
    if prob_recyclabe > 50:
        c2.write(f"Je pense à {prob_recyclabe:.2f} % que l object est recyclabe.")
    else:
        c2.write(f"Je pense à {(prob_organic):.2f} % que l object n'est pas recyclabe.")
