import json
import os

import numpy as np
import streamlit as st
import tensorflow as tf
from huggingface_hub import hf_hub_download
from PIL import Image

REPO_ID = os.environ.get("MODEL_REPO", "danielaxv3/lungcv-xray-classifier")

st.set_page_config(page_title="Chest X-Ray Classification", page_icon=None, layout="wide")

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


try:
    model, config = get_model()
except Exception:
    st.error(f"Model could not be loaded from {REPO_ID}. Check that the HF_TOKEN secret is set in the app settings.")
    st.stop()

order = config["class_order"]
size = tuple(config["image_size"])
version = config.get("model_version", "baseline-v1")

EXAMPLES = [
    ("Normal (PA)", "examples/normal_pa.jpg"),
    ("Lobar pneumonia", "examples/pneumonia_lobar.jpg"),
    ("Influenza pneumonia", "examples/pneumonia_influenza.jpg"),
]

with st.sidebar:
    st.header("Model info")
    st.write(f"**Version:** {version}")
    st.write(f"**Classes:** {', '.join(order)}")
    st.write(f"**Input:** {size[0]}x{size[1]} RGB")
    st.write("EfficientNetB0 transfer learning, trained with patient-wise splits on public chest X-ray data.")
    st.divider()
    st.subheader("Example image credits")
    st.caption("Wikimedia Commons - Normal PA chest radiograph, X-ray of lobar pneumonia, and Chest X-ray in influenza and Haemophilus influenzae by Mikael Haggstrom (CC0).")
    st.divider()
    st.caption("Research prototype. Not for clinical use.")

image_source = None
source_label = None

uploaded = st.file_uploader("Upload a chest X-ray (png / jpg)", type=["png", "jpg", "jpeg"])
if uploaded is not None:
    image_source = uploaded
    source_label = "Uploaded X-ray"

st.subheader("Or test with an example")
cols = st.columns(len(EXAMPLES))
for (name, path), col in zip(EXAMPLES, cols):
    if col.button(name, use_container_width=True):
        st.session_state["example"] = path
        st.session_state.pop("uploaded_key", None)

if image_source is None and st.session_state.get("example"):
    image_source = st.session_state["example"]
    source_label = "Example image (Wikimedia Commons, CC0)"

left, right = st.columns([1, 1], gap="medium")

with left:
    if image_source is not None:
        st.image(image_source, caption=source_label, use_container_width=True)
    else:
        st.info("Upload an X-ray or pick an example to get a prediction. Images are processed in memory only - nothing is stored.")

with right:
    if image_source is not None:
        with st.spinner("Running inference..."):
            img = Image.open(image_source).convert("RGB")
            _w, _h = img.size
            _m = int(min(_w, _h) * 0.04)
            img = img.crop((_m, _m, _w - _m, _h - _m)).resize(size, Image.LANCZOS)
            x = np.asarray(img, dtype=np.float32)[None]
            probs = model.predict(x, verbose=0)[0]
        result = {c: round(float(p), 4) for c, p in zip(order, probs)}
        best = max(result, key=result.get)
        conf = result[best]

        st.subheader(f"Prediction: {best}")
        st.metric("Confidence", f"{conf * 100:.1f}%")

        st.subheader("Class probabilities")
        for c in order:
            p = result[c]
            color = "#16a34a" if c == best else "#94a3b8"
            st.markdown(
                f"<div style='display:flex;align-items:center;gap:8px'>"
                f"<div style='width:110px;font-weight:600'>{c}</div>"
                f"<div style='flex:1;background:#e2e8f0;border-radius:4px;height:20px'>"
                f"<div style='width:{p * 100:.1f}%;height:100%;background:{color};border-radius:4px'></div>"
                f"</div><div style='width:60px;text-align:right'>{p * 100:.1f}%</div></div>",
                unsafe_allow_html=True,
            )

        with st.expander("Raw output (JSON)"):
            st.json(result)
    else:
        st.info("Waiting for an image.")
