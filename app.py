
import streamlit as st
import torch
import timm
import pandas as pd

from PIL import Image
from timm.data import resolve_model_data_config, create_transform

# -----------------------------
# SEITENEINSTELLUNGEN
# -----------------------------

st.set_page_config(
    page_title="Drone Vision AI",
    page_icon="🛰️",
    layout="wide"
)

# -----------------------------
# DESIGN
# -----------------------------

st.markdown("""
<style>
.stApp {
    background-color: #0b111a;
    color: white;
}
h1, h2, h3 {
    color: #e5edf7;
}
section[data-testid="stSidebar"] {
    background-color: #111c29;
}
div[data-testid="stMetric"] {
    background-color: #152334;
    padding: 18px;
    border-radius: 12px;
}
</style>
""", unsafe_allow_html=True)

# -----------------------------
# MODELL LADEN
# -----------------------------

@st.cache_resource
def load_model():

    model = timm.create_model(
        "hf_hub:cm93/resnet50-eurosat",
        pretrained=True
    )

    model.eval()

    config = resolve_model_data_config(model)
    transform = create_transform(
        **config,
        is_training=False
    )

    return model, transform


# -----------------------------
# KATEGORIEN
# -----------------------------

classes = [
    "Annual Crop",
    "Forest",
    "Herbaceous Vegetation",
    "Highway",
    "Industrial",
    "Pasture",
    "Permanent Crop",
    "Residential",
    "River",
    "Sea / Lake"
]

# -----------------------------
# HAUPTSEITE
# -----------------------------

st.title("🛰️ DRONE VISION AI")

st.caption("SATELLITE IMAGE CLASSIFICATION SYSTEM")

st.divider()

st.subheader("Image Analysis")

uploaded_file = st.file_uploader(
    "Upload a satellite image",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(
            image,
            caption="Uploaded Image",
            use_container_width=True
        )

    with col2:

        st.subheader("Analysis")

        if st.button("START ANALYSIS", use_container_width=True):

            with st.spinner("AI is analyzing the image..."):

                model, transform = load_model()

                input_tensor = transform(image).unsqueeze(0)

                with torch.inference_mode():
                    output = model(input_tensor)
                    probabilities = torch.softmax(
                        output, dim=1
                    )[0]

                scores = probabilities.cpu().numpy()

                results = sorted(
                    zip(classes, scores),
                    key=lambda x: x[1],
                    reverse=True
                )

                best_class, best_score = results[0]

                st.success("Analysis completed")

                st.metric(
                    "Predicted Category",
                    best_class
                )

                st.metric(
                    "Confidence",
                    f"{best_score * 100:.2f}%"
                )

                st.divider()

                st.subheader("All Categories")

                df = pd.DataFrame(
                    results,
                    columns=["Category", "Probability"]
                )

                df["Probability"] = (
                    df["Probability"] * 100
                ).round(2)

                st.dataframe(
                    df,
                    use_container_width=True,
                    hide_index=True
                )

                st.bar_chart(
                    df.set_index("Category")
                )

else:
    st.info("Please upload a satellite image to begin.")
