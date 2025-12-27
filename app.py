import streamlit as st
import numpy as np
from PIL import Image
from ultralytics import YOLO
from pathlib import Path
import time
import os
os.environ["ULTRALYTICS_NO_CV2"] = "1"



st.set_page_config(
    page_title="AI Simple Defect Classifier",
    layout="centered"
)

MODEL_PATH = Path(__file__).parent / "best (1).pt"

@st.cache_resource(show_spinner=False)
def load_model():
    return YOLO(str(MODEL_PATH))

model = load_model()


st.title("AI Simple Defect Classifier")
st.markdown(
    "Upload an image or capture a photo to classify an industrial part as **GOOD** or **BAD**."
)

option = st.radio(
    "Choose input method:",
    ["Upload Image", "Take Photo"],
    horizontal=True
)

image = None

if option == "Upload Image":
    uploaded_file = st.file_uploader(
        "Upload an image",
        type=["jpg", "jpeg", "png"]
    )
    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")

elif option == "Take Photo":
    camera_input = st.camera_input("Take a photo")
    if camera_input:
        image = Image.open(camera_input).convert("RGB")

if image is not None:
    st.image(image, caption="Input Image", use_column_width=True)

    with st.spinner("Running AI inspection..."):
        img_np = np.array(image)

        start_time = time.perf_counter()
        results = model.predict(img_np, verbose=False)
        elapsed_ms = (time.perf_counter() - start_time) * 1000

        result = results[0]

        if result.probs is None:
            st.error("Model did not return classification probabilities.")
        else:
            probs = result.probs.data.cpu().numpy()
            pred_class = int(np.argmax(probs))
            confidence = float(probs[pred_class]) * 100

            names = result.names
            label = names[pred_class].upper()

   
    if label == "GOOD":
        st.success(f"GOOD\nConfidence: {confidence:.2f}%")
    else:
        st.error(f"BAD\nConfidence: {confidence:.2f}%")

    st.caption(f"Inference time: {elapsed_ms:.1f} ms")
