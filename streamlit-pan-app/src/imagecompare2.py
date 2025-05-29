import cv2
import os
import matplotlib.pyplot as plt
import time

database_folder = "C:\\Users\\pranjal.prabhu\\Desktop\\image\\archive\\images-high-res"
uploaded_image_path = "C:\\Users\\pranjal.prabhu\\Pictures\\image2.PNG"

def detect_and_crop_face(image_path):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not read image: {image_path}")
        return None
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if len(faces) == 0:
        print(f"No face detected in: {image_path}")
    for (x, y, w, h) in faces:
        return gray[y:y+h, x:x+w]
    return None

def get_orb_descriptor(face_img):
    if face_img is None:
        return None
    orb = cv2.ORB_create(nfeatures=1000)
    face_img = cv2.resize(face_img, (128, 128))
    face_img = cv2.equalizeHist(face_img)
    keypoints, descriptors = orb.detectAndCompute(face_img, None)
    if descriptors is None:
        print("No descriptors found.")
    return descriptors

def find_matching_pancard(uploaded_image_path, database_folder):
    uploaded_face = detect_and_crop_face(uploaded_image_path)
    if uploaded_face is None:
        print("No face found in uploaded image.")
        return None
    uploaded_desc = get_orb_descriptor(uploaded_face)
    if uploaded_desc is None:
        print("No descriptors found in uploaded image.")
        return None
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    best_match = None
    best_score = float('inf')
    for filename in os.listdir(database_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png')):
            db_image_path = os.path.join(database_folder, filename)
            db_face = detect_and_crop_face(db_image_path)
            if db_face is not None:
                db_desc = get_orb_descriptor(db_face)
                if db_desc is not None:
                    matches = bf.match(uploaded_desc, db_desc)
                    print(f"{filename}: {len(matches)} matches")
                    score = sum([m.distance for m in matches]) / (len(matches) + 1e-6)
                    if score < best_score and len(matches) > 5:  # Lowered threshold for testing
                        best_score = score
                        best_match = db_image_path

    if best_match is not None:
        return best_match, f"Best match: {best_match} (score: {best_score:.2f})"
    return None, "No matching PAN card found in database."

if __name__ == "__main__":
    start_time = time.time()
    result = find_matching_pancard(uploaded_image_path, database_folder)
    end_time = time.time()
    print(f"Time taken: {end_time - start_time:.2f} seconds")
    if result:
        matched_image = cv2.imread(result)
        plt.imshow(cv2.cvtColor(matched_image, cv2.COLOR_BGR2RGB))
        plt.title("Matched PAN Card")
        plt.axis("off")
        plt.show()