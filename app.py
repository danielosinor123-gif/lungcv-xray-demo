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
uncertainty_threshold = float(config.get("uncertainty_threshold", 0.60))

EXAMPLES = [
    ("NORMAL sample", "examples/normal_indist.jpg"),
    ("PNEUMONIA sample", "examples/pneumonia_indist.jpg"),
    ("COVID sample", "examples/covid_indist.jpg"),
    ("TB sample", "examples/tb_indist.jpg"),
]

with st.sidebar:
    st.header("Model info")
    st.write(f"**Version:** {version}")
    st.write(f"**Classes:** {', '.join(order)}")
    st.write(f"**Input:** {size[0]}x{size[1]} RGB")
    st.write("EfficientNetB0 transfer learning, trained with patient-wise splits on public chest X-ray data.")
    st.divider()
    st.subheader("Example image credits")
    st.caption("Wikimedia Commons - Normal PA chest radiograph, X-ray of lobar pneumonia, Example images are samples from the public training datasets (NIH, Kermany, COVID-Radiography, QU-Ex, TB sets) - see dataset credits in the model card.")
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
            _crop = config.get("crop_borders", 0)
            if isinstance(_crop, str):
                import re as _re
                _mm = _re.search(r"([\d.]+)\s*%", _crop)
                _crop = (float(_mm.group(1)) / 100.0) if _mm else 0.0
            _w, _h = img.size
            _m = int(min(_w, _h) * float(_crop or 0))
            if _m > 0:
                img = img.crop((_m, _m, _w - _m, _h - _m))
            img = img.resize(size, Image.LANCZOS)
            x = np.asarray(img, dtype=np.float32)[None]
            raw = model.predict(x, verbose=0)
        probs = raw[0] if isinstance(raw, list) else raw
        if getattr(probs, 'ndim', 1) == 2:
            probs = probs[0]
        result = {c: round(float(p), 4) for c, p in zip(order, probs)}
        best = max(result, key=result.get)
        conf = result[best]

        ranked = sorted(result.items(), key=lambda t: -t[1])
        runner_up = ranked[1] if len(ranked) > 1 else (None, 0.0)
        spread = conf - runner_up[1]
        uncertain = conf < uncertainty_threshold

        if uncertain:
            st.warning("Low confidence - the model cannot classify this film reliably. Radiologist review is recommended. The full distribution is shown below for transparency.")
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

        st.subheader("Findings summary (auto-generated)")
        summary = [
            f"The model assigns **{conf * 100:.1f}% confidence** to the class **{best}**.",
            f"The next most likely class is {runner_up[0]} at {runner_up[1] * 100:.1f}% (separation: {spread * 100:.1f} points).",
        ]
        if uncertain:
            summary.append("Confidence is below the reporting threshold - these findings are **inconclusive**, and a radiologist review is recommended.")
        else:
            summary.append("Confidence is above the reporting threshold - the model considers this classification reliable for research purposes.")
        summary.append("These statements describe the model statistical outputs only. They are not clinical observations and must not be used for diagnosis.")
        for s in summary:
            st.markdown("- " + s)
    else:
        st.info("Waiting for an image.")


with st.expander("What is next - the roadmap to the next level"):
    st.markdown("""**1. Fine-tune the encoder** - the frozen EfficientNetB0 base unfrozen on its top blocks (low learning rate) - expected +1-3 points on the weak classes.

**2. External validation with hospital data** - the hospital proposal is ready; partner samples become the locked external test cohort - the industry-standard evidence of generalization.

**3. Language layer v2 - a grounded LLM narrator** - the model calibrated outputs and lung mask fed to a language model with strict grounding (it describes only what the model computed - no invented findings).

**4. Data ladder** - MIMIC-CXR (377k images, report-mined labels) and CheXpert (224k) - the corpora the field trains on; the pipeline absorbs them like the current seven datasets.

**5. DICOM ingestion + triage framing** - hospital-workflow integration: results flagged for radiologist review, never autonomous diagnosis.

**6. Drift monitoring** - per-scanner performance tracking over time - the layer where deployed models silently fail without it.

Research prototype only - not a medical diagnosis.
""")