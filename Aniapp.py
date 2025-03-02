import streamlit as st
import requests
from PIL import Image

st.title("🐾 Animal Classifier (FastAPI Backend)")

# Upload user image
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "png", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Send to FastAPI
    files = {"file": uploaded_file.getvalue()}
    response = requests.post("http://127.0.0.1:8000/predict/", files=files)

    if response.status_code == 200:
        result = response.json()
        st.subheader(f"Predicted Animal: **{result['Predicted Animal']}**")
    else:
        st.error(response.json()["detail"])

# Load a random dataset image for testing
if st.button("Test with Hugging Face Dataset"):
    response = requests.get("http://127.0.0.1:8000/dataset-sample/")
    if response.status_code == 200:
        label = response.json()["sample_label"]
        st.write(f"Random Dataset Sample Label: **{label}**")
    else:
        st.error("Failed to load dataset sample.")
