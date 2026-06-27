"""
BraTS 2020 Brain Tumor Segmentation — Streamlit Demo
Upload an H5 slice → model predicts → see results
"""

import os
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt
from backend import load_model, load_h5, preprocess, predict, compute_metrics
from backend import make_vis_image, make_figure

# ─────────────────────────────────────────────────────────────────────────────
# Page setup
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="Brain Tumor Segmentation", page_icon="🧠", layout="wide")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;800&family=JetBrains+Mono:wght@400;600&display=swap');
    .metric-card {
        background: linear-gradient(145deg, #141c2f, #111827);
        border-radius: 12px; padding: 1rem 1.2rem;
        border: 1px solid #1e293b; text-align: center;
    }
    .metric-card:hover { border-color: #3b82f6; }
    .metric-value { font-family: 'JetBrains Mono', monospace; font-size: 1.8rem; font-weight: 700; color: #60a5fa; }
    .metric-label { font-size: 0.75rem; color: #64748b; text-transform: uppercase; letter-spacing: 0.06em; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar — settings
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## Settings")
    ckpt_path = st.text_input("Model checkpoint path", value="SwinUNet_best.pt")
    threshold = st.slider("Prediction threshold", 0.1, 0.9, 0.4, 0.05)
    device = "cpu"
    st.caption(f"Running on: **{device.upper()}**")


# ─────────────────────────────────────────────────────────────────────────────
# Load model (cached)
# ─────────────────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner="Loading model…")
def _load(path, dev):
    return load_model(path, dev)

model = None
if os.path.isfile(ckpt_path):
    try:
        model = _load(ckpt_path, device)
        st.sidebar.success("Model loaded ✓")
    except Exception as e:
        st.sidebar.error(f"Failed to load: {e}")
else:
    st.sidebar.warning(f"Checkpoint not found: `{ckpt_path}`")


# ─────────────────────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("# 🧠 Brain Tumor Segmentation")
st.markdown("Upload a BraTS `.h5` slice — the model segments the tumor and shows results.")


# ─────────────────────────────────────────────────────────────────────────────
# Upload + Predict
# ─────────────────────────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload H5 slice", type=["h5", "hdf5"])

if uploaded is not None:
    tmp = f"temp_{uploaded.name}"
    with open(tmp, "wb") as f:
        f.write(uploaded.read())

    img_hwc, gt_mask = load_h5(tmp)
    tensor = preprocess(img_hwc)
    img_vis = make_vis_image(tensor[0])

    # Resize gt_mask to match model output (IMG_SIZE × IMG_SIZE)
    from skimage.transform import resize as sk_resize
    if gt_mask.shape != (256, 256):
        gt_mask = (sk_resize(gt_mask.astype(np.float32), (256, 256), order=0,
                             preserve_range=True) > 0.5).astype(np.uint8)

    if model is not None:
        with st.spinner("Running inference…"):
            prob_map, pred_mask = predict(model, tensor, device, threshold)
            metrics = compute_metrics(pred_mask, gt_mask)

        # Metric cards
        c1, c2, c3, c4 = st.columns(4)
        for col, label, key in [
            (c1, "Dice Score",  "dice"),
            (c2, "Sensitivity", "sensitivity"),
            (c3, "Precision",   "precision"),
            (c4, "IoU",         "iou"),
        ]:
            col.markdown(
                f'<div class="metric-card">'
                f'<div class="metric-value">{metrics[key]:.3f}</div>'
                f'<div class="metric-label">{label}</div></div>',
                unsafe_allow_html=True,
            )
        st.markdown("")

        # 3-panel figure
        fig = make_figure(img_vis, pred_mask)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

    else:
        st.warning("No model loaded — showing raw MRI only.")
        fig, ax = plt.subplots(1, 1, figsize=(5, 4))
        fig.patch.set_facecolor("#0a0e1a")
        ax.imshow(img_vis, cmap="gray"); ax.set_title("MRI", color="#94a3b8"); ax.axis("off")
        plt.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
