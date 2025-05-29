import streamlit as st
import cv2
import numpy as np
import os
import tempfile
from ultralytics import YOLO
import easyocr

from finalpan import detect_pan_card_fields
from imagecompare2 import find_matching_pancard

@st.cache_resource
def load_yolo_model(model_path):
    return YOLO(model_path)

@st.cache_resource
def load_easyocr_reader(langs):
    return easyocr.Reader(langs)

def main():
    st.title("PAN Card Detection & Face Matching")

    # --- Model Update Section ---
    st.sidebar.header("Model Settings")
    default_model_path = 'model/best.pt'
    uploaded_model = st.sidebar.file_uploader("Upload new YOLO model (.pt)", type=["pt"])
    if uploaded_model is not None:
        temp_model_path = os.path.join(tempfile.gettempdir(), uploaded_model.name)
        with open(temp_model_path, "wb") as f:
            f.write(uploaded_model.read())
        model_path = temp_model_path
        st.sidebar.success(f"Loaded custom model: {uploaded_model.name}")
    else:
        model_path = default_model_path

    langs = st.sidebar.multiselect("EasyOCR Languages", ["en", "hi"], default=["en"])
    model = load_yolo_model(model_path)
    reader = load_easyocr_reader(langs)

    # --- PAN Card Detection ---
    st.header("1. PAN Card Detection and OCR")
    uploaded_file = st.file_uploader("Upload a PAN card image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        annotated_image, pan_number, name = detect_pan_card_fields(image)
        st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), caption='Detected Fields', use_column_width=True)
        st.write(f"**PAN Number:** {pan_number}")
        st.write(f"**Name:** {name}")

    # --- Face Matching ---
    st.header("2. Find Matching PAN Card by Face")
    uploaded_face_file = st.file_uploader("Upload a PAN card image (for face matching)", type=["jpg", "jpeg", "png"], key="face")
    if uploaded_face_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_face_file.read()), dtype=np.uint8)
        face_image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        import time
        # database_folder = "C:\\Users\\pranjal.prabhu\\Desktop\\streamlit_2\\archive\\images-high-res"
        database_folder = 'archive\\images-high-res'
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            cv2.imwrite(tmp.name, face_image)
            temp_face_path = tmp.name
        with st.spinner("Searching for best match in database..."):
            start_time = time.time()
            match_img, match_msg = find_matching_pancard(temp_face_path, database_folder)
            end_time = time.time()
            elapsed = end_time - start_time
        st.write(match_msg)
        st.write(f"Time taken: {elapsed:.2f} seconds")
        if match_img is not None:
            if match_img is not None and isinstance(match_img, str):
                img = cv2.imread(match_img)
                if img is not None:
                    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption='Best Match from Database', use_column_width=True)
                else:
                    st.warning("Matched image could not be loaded.")            
if __name__ == "__main__":
    main()
