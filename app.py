import json
import os

import numpy as np
import streamlit as st
import tensorflow as tf
from huggingface_hub import hf_hub_download
from PIL import Image

REPO_ID = os.environ.get("MODEL_REPO", "danielaxv3/lungcv-xray-classifier")

st.set_page_config(page_title="Chest X-Ray Classification")

st.title("Chest X-Ray Classification")
st.warning("Research prototype only. This output is not a medical diagnosis.")


@st.cache_resource
def get_model():
    token = st.secrets.get("HF_TOKEN")
    model_path = hf_hub_download(repo_id=REPO_ID, filename="best_model.keras", token=token)
    config_path = hf_hub_download(repo_id=REPO_ID, filename="preprocessing_config.json", token=token)
    model = tf.keras.models.load_model(model_path)
    with open(config_path) as f:
        config = json.load(f)
    return model, config


model, config = get_model()
order = config["class_order"]
size = tuple(config["image_size"])

uploaded = st.file_uploader("Upload a chest X-ray", type=["png", "jpg", "jpeg"])
if uploaded is not None:
    img = Image.open(uploaded).convert("RGB").resize(size, Image.LANCZOS)
    x = np.asarray(img, dtype=np.float32)[None]
    probs = model.predict(x, verbose=0)[0]
    result = {c: round(float(p), 4) for c, p in zip(order, probs)}
    st.subheader(f"Prediction: {max(result, key=result.get)}")
    st.bar_chart({c: p for c, p in result.items()})
    st.json(result)
