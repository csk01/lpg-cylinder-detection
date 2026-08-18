import streamlit as st
from PIL import Image
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from predict import load_models, predict

st.set_page_config(
    page_title="LPG Cylinder Identification",
    page_icon="🛢",
    layout="centered"
)

# Load models once — cached across sessions
@st.cache_resource
def get_models():
    return load_models()

detector, classifier, device = get_models()

# ── UI ────────────────────────────────────────────────────────────────────
st.title("LPG Cylinder Identification")
st.caption("YOLOv11x detection → EfficientNetB2 classification")

uploaded = st.file_uploader(
    "Upload a cylinder image",
    type=["jpg", "jpeg", "png", "webp"]
)

if uploaded:
    img = Image.open(uploaded).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:
        st.image(img, caption="Input image", use_container_width=True)

    with st.spinner("Detecting and classifying..."):
        result = predict(img, detector, classifier, device)

    if result["status"] == "no_cylinder":
        st.error("No LPG cylinder detected in this image.")

    elif result["status"] == "multi_cylinder":
        st.warning("Multiple cylinders detected. Please present one cylinder at a time.")

    else:
        with col2:
            st.image(result["crop"], caption="Detected cylinder crop",
                     use_container_width=True)

        # Brand result
        brand = result["brand"].title()
        conf  = result["confidence"]

        st.markdown(f"### Brand: **{brand}**")
        st.markdown(f"Confidence: **{conf}%**")

        # Probability bars
        st.markdown("**All probabilities:**")
        for b, p in result["probabilities"].items():
            st.progress(int(p), text=f"{b.title()}: {p}%")

        # Flag wrong prediction
        with st.expander("Flag wrong prediction"):
            reason = st.selectbox("Reason", [
                "Wrong brand",
                "Not detected correctly",
                "Multiple cylinders",
                "Other"
            ])
            if st.button("Submit flag"):
                img.save(f"flagged_predictions/{uploaded.name}")
                st.success("Flagged — thank you!")