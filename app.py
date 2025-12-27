import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO
from pathlib import Path
import time

# Load YOLOv8 model (cached) with robust Windows path handling
MODEL_PATH = Path(__file__).parent / "best (1).pt"

@st.cache_resource(show_spinner=False)
def load_model():
    try:
        return YOLO(str(MODEL_PATH))
    except Exception as e:
        st.error(f"Failed to load model from '{MODEL_PATH}': {e}")
        raise

model = load_model()

st.set_page_config(page_title="AI Defect Classifier", layout="centered")

# UI Title
st.title("🔍 AI Simple Defect Classifier")
st.markdown("Upload an image or capture a photo to classify industrial parts.")

# Image input
option = st.radio("Choose input method:", ["Upload Image", "Take Photo"])

image = None

if option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
    if uploaded_file:
        image = Image.open(uploaded_file)

elif option == "Take Photo":
    camera_input = st.camera_input("Take a photo")
    if camera_input:
        image = Image.open(camera_input)

# Inference
if image is not None:
    st.image(image, caption="Input Image", use_column_width=True)

    with st.spinner("Running AI inspection..."):
        img_np = np.array(image)
        img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)

        start_t = time.perf_counter()
        results = model(img_np)[0]
        elapsed_ms = (time.perf_counter() - start_t) * 1000.0

        if getattr(results, "probs", None) is None:
            st.error("Model did not return classification probabilities.")
        else:
            probs = results.probs.data.cpu().numpy()
            confidence = float(np.max(probs)) * 100
            pred_class = int(np.argmax(probs))

            # Best class name
            names = getattr(results, "names", {})
            best_label = names.get(pred_class, str(pred_class))

            # Attempt to read model image size (fallback to 224x224)
            try:
                imgsz = getattr(model, "model", None)
                imgsz = getattr(imgsz, "args", {}).get("imgsz", 224)
            except Exception:
                imgsz = 224
            if isinstance(imgsz, (list, tuple)) and len(imgsz) >= 2:
                shape_str = f"{imgsz[1]}x{imgsz[0]}"
            else:
                shape_str = f"{int(imgsz)}x{int(imgsz)}"

            # Compose full probabilities line in requested format
            try:
                items = []
                for i in range(len(probs)):
                    name_i = names.get(i, str(i))
                    items.append(f"{name_i} {probs[i]:.2f}")
                items_str = ", ".join(items)
                st.text(f"0: {shape_str} {items_str}, {elapsed_ms:.1f}ms")
            except Exception:
                # Fallback minimal line
                st.text(f"0: {shape_str} top='{best_label}' {confidence/100.0:.2f}, {elapsed_ms:.1f}ms")

    # Output display
    if getattr(results, "probs", None) is not None:
        st.success(f"Top class: {best_label}\nConfidence: {confidence:.2f}%")
