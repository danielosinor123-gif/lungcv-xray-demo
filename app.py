import json
import os

import numpy as np
import streamlit as st
import tensorflow as tf
from huggingface_hub import hf_hub_download
from PIL import Image

REPO_ID = os.environ.get("MODEL_REPO", "danielaxv3/lungcv-xray-multilabel-v3")

st.set_page_config(page_title="Chest X-Ray Finding Scores", page_icon=None, layout="wide")

st.title("Chest X-Ray Finding Scores")
st.warning("Research prototype only. This output is not a medical diagnosis.")


@st.cache_resource
def get_model():
    token = st.secrets.get("HF_TOKEN")
    model_path = hf_hub_download(repo_id=REPO_ID, filename="best_model.keras", token=token)
    config_path = hf_hub_download(repo_id=REPO_ID, filename="preprocessing_config.json", token=token)
    model = tf.keras.models.load_model(model_path)
    with open(config_path) as f:
        config = json.load(f)
    # OOD detector artifacts (present on multilabel-v3 and later)
    centroids = None
    try:
        cent_path = hf_hub_download(repo_id=REPO_ID, filename="ood_centroids.json", token=token)
        with open(cent_path) as f:
            centroids = json.load(f)
    except Exception:
        centroids = None
    feat_model = None
    if centroids is not None:
        try:
            feat_model = tf.keras.Model(model.input, model.get_layer("gap").output)
        except Exception:
            feat_model = None
    return model, feat_model, centroids, config


try:
    model, feat_model, centroids, config = get_model()
except Exception:
    st.error(f"Model could not be loaded from {REPO_ID}. Check that the HF_TOKEN secret is set in the app settings.")
    st.stop()

order = config["class_order"]
size = tuple(config["image_size"])
version = config.get("model_version", "baseline-v1")
uncertainty_threshold = float(config.get("uncertainty_threshold", 0.5))
clahe_applied = bool(config.get("clahe_applied", False))
_ood_thr = config.get("ood_distance_threshold")
ood_threshold = float(_ood_thr) if _ood_thr is not None else None
norm_method = str(config.get("normalization_method", "")).lower()

EXAMPLES = [
    ("NORMAL sample", "examples/normal_indist.jpg"),
    ("PNEUMONIA sample", "examples/pneumonia_indist.jpg"),
    ("COVID sample", "examples/covid_indist.jpg"),
    ("TB sample", "examples/tb_indist.jpg"),
]


def preprocess(img, cfg):
    """Replicates the training preprocessing contract exactly (config-driven).

    crop (if any) -> grayscale -> CLAHE (if enabled) -> resize -> 3-channel
    -> model-specific normalization (from the config's normalization_method).
    """
    _crop = cfg.get("crop_borders", 0)
    if isinstance(_crop, str):
        import re as _re

        _mm = _re.search(r"([\d.]+)\s*%", _crop)
        _crop = (float(_mm.group(1)) / 100.0) if _mm else 0.0
    _w, _h = img.size
    _m = int(min(_w, _h) * float(_crop or 0))
    if _m > 0:
        img = img.crop((_m, _m, _w - _m, _h - _m))
    gray = img.convert("L")
    if bool(cfg.get("clahe_applied", False)):
        import cv2

        arr8 = np.asarray(gray)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        arr8 = clahe.apply(arr8)
        gray = Image.fromarray(arr8)
    img_out = gray.resize(tuple(cfg["image_size"]), Image.LANCZOS)
    arr = np.asarray(img_out, dtype=np.float32)
    x = np.stack([arr, arr, arr], axis=-1)[None]  # (1, H, W, 3) in [0, 255]
    method = str(cfg.get("normalization_method", "")).lower()
    if "densenet" in method and "preprocess_input" in method:
        x = tf.keras.applications.densenet.preprocess_input(x)
    # other known methods ("raw_0_255", legacy EfficientNet) take raw [0, 255]
    return x


def ood_distance(x):
    """Distance to the nearest class centroid in the GAP feature space."""
    if feat_model is None or centroids is None or not ood_threshold:
        return None, None
    try:
        feat = feat_model.predict(x, verbose=0)[0]
    except Exception:
        return None, None
    dists = {}
    for k, v in centroids.items():
        idx = int(k)
        if 0 <= idx < len(order):
            dists[order[idx]] = float(np.linalg.norm(feat - np.asarray(v, dtype=np.float32)))
    if not dists:
        return None, None
    nearest = min(dists, key=dists.get)
    return dists[nearest], nearest


with st.sidebar:
    st.header("Model info")
    st.write(f"**Version:** {version}")
    st.write(f"**Findings:** {', '.join(order)}")
    st.write(f"**Input:** {size[0]}x{size[1]} grayscale + CLAHE" if clahe_applied else f"**Input:** {size[0]}x{size[1]} RGB")
    st.write("DenseNet121 transfer learning with an independent sigmoid score per finding (multi-label). Films below the reporting threshold are flagged uncertain; an OOD detector flags unfamiliar films.")
    if centroids is not None and ood_threshold:
        st.write(f"**OOD detector:** on (threshold {ood_threshold:.1f})")
    st.divider()
    st.subheader("Example image credits")
    st.caption("Example images are samples from the public training datasets (NIH, Kermany, COVID-Radiography, QU-Ex, TB sets) - see dataset credits in the model card.")
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
    source_label = "Example image (from the public training data)"

left, right = st.columns([1, 1], gap="medium")

with left:
    if image_source is not None:
        st.image(image_source, caption=source_label, use_container_width=True)
    else:
        st.info("Upload an X-ray or pick an example to get the finding scores. Images are processed in memory only - nothing is stored.")

with right:
    if image_source is not None:
        with st.spinner("Running inference..."):
            img = Image.open(image_source)
            x = preprocess(img, config)
            raw = model.predict(x, verbose=0)
            dist, nearest = ood_distance(x)
        probs = raw[0] if isinstance(raw, list) else raw
        if getattr(probs, "ndim", 1) == 2:
            probs = probs[0]
        seg_mask = None
        if isinstance(raw, list) and len(raw) > 1:
            seg_mask = np.asarray(raw[1][0, :, :, 0], dtype=np.float32)
        result = {c: round(float(p), 4) for c, p in zip(order, probs)}
        best = max(result, key=result.get)
        conf = result[best]
        present = sorted(((c, result[c]) for c in order if result[c] >= uncertainty_threshold), key=lambda t: -t[1])
        uncertain = conf < uncertainty_threshold
        ood_flag = dist is not None and dist > ood_threshold

        if ood_flag:
            st.error(
                f"**Out-of-distribution film** - this image does not resemble the training data "
                f"(nearest-class distance {dist:.1f} vs threshold {ood_threshold:.1f}). "
                f"The finding scores below should not be trusted for this film. Radiologist review is recommended."
            )
        elif dist is not None:
            st.caption(f"In-distribution check passed: nearest-class distance {dist:.1f} (threshold {ood_threshold:.1f}).")

        if uncertain and not ood_flag:
            st.warning("Low confidence - no finding crosses the reporting threshold, so the model cannot classify this film reliably. Radiologist review is recommended. The full scores are shown below for transparency.")

        st.subheader(f"Highest finding: {best}")
        st.metric("Score", f"{conf * 100:.1f}%")

        st.subheader("Finding scores (independent)")
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
        st.caption("Scores are independent sigmoid outputs, not a probability distribution - several findings can be high at once.")

        st.subheader("Lung region (model attention)")
        if seg_mask is not None:
            base_img = np.asarray(Image.open(image_source).convert("RGB").resize((224, 224), Image.LANCZOS), dtype=np.float32)
            m = seg_mask > 0.5
            overlay = base_img.copy()
            overlay[m] = overlay[m] * 0.45 + np.array([0.0, 200.0, 0.0]) * 0.55
            st.image(overlay.astype(np.uint8), caption="Green: lung regions the model traced (model attention, not clinical segmentation)", use_container_width=True)
        else:
            st.caption("This model has no segmentation head - lung-region overlay unavailable.")

        with st.expander("Raw output (JSON)"):
            payload = dict(result)
            if dist is not None:
                payload["ood_nearest_class"] = nearest
                payload["ood_distance"] = round(dist, 2)
                payload["ood_threshold"] = ood_threshold
            st.json(payload)

        st.subheader("Findings summary (auto-generated)")
        summary = []
        if ood_flag:
            summary.append(
                f"The OOD detector flags this film as **outside the training distribution** "
                f"(nearest-class distance {dist:.1f} exceeds the threshold {ood_threshold:.1f}). "
                f"The scores above are reported for transparency but should **not** be interpreted as findings for this film."
            )
        elif uncertain:
            summary.append(f"No finding crosses the {uncertainty_threshold * 100:.0f}% reporting threshold (highest: {best} at {conf * 100:.1f}%). These results are **inconclusive**, and a radiologist review is recommended.")
        else:
            listing = ", ".join(f"**{c}** ({p * 100:.1f}%)" for c, p in present)
            summary.append(f"The model reports {len(present)} finding(s) above the {uncertainty_threshold * 100:.0f}% reporting threshold: {listing}.")
            below = sorted(((c, result[c]) for c in order if result[c] < uncertainty_threshold), key=lambda t: -t[1])
            if below:
                summary.append("Findings below the threshold (not reported): " + ", ".join(f"{c} ({p * 100:.1f}%)" for c, p in below) + ".")
        summary.append("These statements describe the model's statistical outputs only. They are not clinical observations and must not be used for diagnosis.")
        for s in summary:
            st.markdown("- " + s)
    else:
        st.info("Waiting for an image.")


with st.expander("What is next - the roadmap to the next level"):
    st.markdown("""**1. Backbone fine-tuning** - tested: a RadImageNet medical-pretrained backbone with two-stage fine-tuning was evaluated against the ImageNet baseline and lost (79.0% vs 82.9%) - the comparison is documented in the model card. The ImageNet DenseNet121 remains production.

**2. External validation with hospital data** - the hospital proposal is ready; partner samples become the locked external test cohort - the industry-standard evidence of generalization.

**3. Language layer v2 - a grounded LLM narrator** - the model's finding scores and confidence fed to a language model with strict grounding (it describes only what the model computed - no invented findings).

**4. Data ladder** - MIMIC-CXR (377k images, report-mined labels) and CheXpert (224k) - the corpora the field trains on; the pipeline absorbs them like the current seven datasets.

**5. DICOM ingestion + triage framing** - hospital-workflow integration: results flagged for radiologist review, never autonomous diagnosis.

**6. Drift monitoring** - per-scanner performance tracking over time - the layer where deployed models silently fail without it.

Research prototype only - not a medical diagnosis.
""")
