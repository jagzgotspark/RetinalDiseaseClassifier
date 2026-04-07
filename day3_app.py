"""
Day 3: Streamlit demo app
Run: streamlit run day3_app.py

Upload a fundus image → get DR grade prediction + Grad-CAM heatmap.
Requires best_model.pth from Day 2.
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from PIL import Image
import io

import torch
import torchvision.transforms as transforms
import streamlit as st

from day1_setup import build_model, GRADE_LABELS
from day2_train import GradCAM, overlay_heatmap

# ── Config ────────────────────────────────────────────────────────────────────
CHECKPOINT_PATH = "best_model.pth"
IMG_SIZE = 224
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

GRADE_COLORS = {
    0: "#2ecc71",   # green — No DR
    1: "#f1c40f",   # yellow — Mild
    2: "#e67e22",   # orange — Moderate
    3: "#e74c3c",   # red — Severe
    4: "#8e44ad",   # purple — Proliferative
}

GRADE_INFO = {
    0: "No signs of diabetic retinopathy detected.",
    1: "Mild NPDR — microaneurysms present. Monitor regularly.",
    2: "Moderate NPDR — more than just microaneurysms. Refer to ophthalmologist.",
    3: "Severe NPDR — any of 4-2-1 rule criteria met. Urgent referral needed.",
    4: "Proliferative DR — neovascularization present. Immediate treatment required.",
}


# ── Model loading (cached) ────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model = build_model(num_classes=5)
    model.load_state_dict(torch.load(CHECKPOINT_PATH, map_location=device))
    model.to(device)
    model.eval()
    return model


def preprocess(img_pil):
    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                              [0.229, 0.224, 0.225]),
    ])
    return transform(img_pil).unsqueeze(0).to(device)


def predict_and_explain(model, img_pil):
    img_tensor = preprocess(img_pil)
    gradcam = GradCAM(model)

    with torch.enable_grad():
        cam, pred_class = gradcam.generate(img_tensor)

    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1).squeeze().cpu().numpy()

    overlay = overlay_heatmap(img_pil, cam, alpha=0.5)
    return pred_class, probs, overlay, cam


def fig_to_bytes(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf


# ── Streamlit UI ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Retinal DR Classifier",
    page_icon="👁️",
    layout="wide",
)

st.title("Retinal Disease Classifier")
st.caption("Diabetic Retinopathy Grading with Grad-CAM Explainability · EfficientNet-B0 · APTOS 2019")

st.markdown("""
Upload a fundus photograph to get:
- **DR grade** (0–4 scale used clinically)
- **Confidence scores** per class
- **Grad-CAM heatmap** showing which retinal regions influenced the prediction
""")

uploaded = st.file_uploader("Upload fundus image (.jpg or .png)", type=["jpg", "jpeg", "png"])

if uploaded is not None:
    img_pil = Image.open(uploaded).convert("RGB")

    with st.spinner("Running model + Grad-CAM..."):
        model = load_model()
        pred_class, probs, overlay, cam = predict_and_explain(model, img_pil)

    grade_name = GRADE_LABELS[pred_class]
    grade_color = GRADE_COLORS[pred_class]
    grade_info = GRADE_INFO[pred_class]

    # ── Grade banner ──────────────────────────────────────────────────────────
    st.markdown(
        f"""<div style="background:{grade_color}22;border-left:4px solid {grade_color};
        padding:12px 18px;border-radius:6px;margin-bottom:1rem;">
        <span style="font-size:20px;font-weight:600;color:{grade_color}">
        Grade {pred_class} — {grade_name}</span><br>
        <span style="color:#555;font-size:14px">{grade_info}</span></div>""",
        unsafe_allow_html=True,
    )

    # ── Two-column layout ─────────────────────────────────────────────────────
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Original image")
        st.image(img_pil.resize((IMG_SIZE, IMG_SIZE)), use_container_width=True)

    with col2:
        st.subheader("Grad-CAM explanation")
        st.image(overlay, use_container_width=True)
        st.caption("Red/yellow = regions with highest influence on prediction")

    # ── Confidence bar chart ──────────────────────────────────────────────────
    st.subheader("Prediction confidence")
    fig, ax = plt.subplots(figsize=(8, 2.5))
    bar_colors = [GRADE_COLORS[i] for i in range(5)]
    bars = ax.barh(
        [f"Grade {i} — {GRADE_LABELS[i]}" for i in range(5)],
        probs * 100,
        color=bar_colors,
        alpha=0.85,
    )
    ax.set_xlim(0, 100)
    ax.set_xlabel("Confidence (%)")
    ax.spines[["top", "right"]].set_visible(False)
    for bar, prob in zip(bars, probs):
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height() / 2,
                f"{prob*100:.1f}%", va="center", fontsize=9)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    # ── Download buttons ──────────────────────────────────────────────────────
    col3, col4 = st.columns(2)
    with col3:
        overlay_pil = Image.fromarray(overlay)
        buf = io.BytesIO()
        overlay_pil.save(buf, format="PNG")
        st.download_button("Download Grad-CAM image", buf.getvalue(),
                           file_name="gradcam_result.png", mime="image/png")
    with col4:
        fig2, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].imshow(img_pil.resize((IMG_SIZE, IMG_SIZE)))
        axes[0].set_title("Original")
        axes[0].axis("off")
        axes[1].imshow(overlay)
        axes[1].set_title(f"Grad-CAM · Pred: Grade {pred_class} ({grade_name})")
        axes[1].axis("off")
        plt.tight_layout()
        report_buf = fig_to_bytes(fig2)
        st.download_button("Download full report", report_buf.getvalue(),
                           file_name="dr_report.png", mime="image/png")
        plt.close()

else:
    st.info("Upload a fundus image above to get started. "
            "Sample images are available in the APTOS 2019 Kaggle dataset.")

    st.markdown("---")
    st.subheader("About this project")
    st.markdown("""
    This tool was built as a research demonstration project in 3 days using:
    - **Model**: EfficientNet-B0 fine-tuned on APTOS 2019 (3,662 labelled fundus images)
    - **Training**: 20 epochs, AdamW optimizer, cosine LR decay
    - **Metric**: Quadratic Weighted Kappa (competition standard)
    - **Explainability**: Grad-CAM on the final convolutional block

    Grad-CAM (Gradient-weighted Class Activation Mapping) shows *which regions*
    of the retina influenced the model's decision — making the black box more transparent,
    which is critical for clinical adoption of AI tools.
    """)