import cv2
from ultralytics import YOLO
import easyocr
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np

def detect_pan_card_fields(image):
    model = YOLO("C:\\Users\\pranjal.prabhu\\Desktop\\image\\runs\\detect\\train\\weights\\best.pt")
    results = model(image)
    boxes = results[0].boxes.xyxy.cpu().numpy()
    class_ids = results[0].boxes.cls.cpu().numpy().astype(int)
    class_names = results[0].names

    reader = easyocr.Reader(['en'])
    annotated_image = image.copy()
    pan_number = "Not found"
    name = "Not found"

    for box, class_id in zip(boxes, class_ids):
        x1, y1, x2, y2 = map(int, box)
        label = class_names[class_id]
        color = (0, 255, 0) if label == "Pan Number" else (255, 0, 0)
        cv2.rectangle(annotated_image, (x1, y1), (x2, y2), color, 2)
        cv2.putText(annotated_image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        crop = image[y1:y2, x1:x2]
        ocr_result = reader.readtext(crop, detail=0)
        if label.lower() == "pan number" and ocr_result:
            pan_number = " ".join(ocr_result)
        if label.lower() == "name" and ocr_result:
            name = " ".join(ocr_result)

    return annotated_image, pan_number, name

def main():
    st.title("PAN Card Field Detection")
    uploaded_file = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = np.array(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

        annotated_image, pan_number, name = detect_pan_card_fields(image)

        st.image(cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB), caption="Annotated Image", use_column_width=True)
        st.write(f"PAN Number: {pan_number}")
        st.write(f"Name: {name}")

if __name__ == "__main__":
    main()