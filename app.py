import streamlit as st
import numpy as np
import cv2
import mediapipe as mp
import joblib
from PIL import Image
import tempfile
import os
from ultralytics import YOLO

st.set_page_config(page_title="Engagement Detection", page_icon="📈", layout="wide")

# ------------- CONFIG -------------
HAND_RAISE_THRESHOLD = -0.05
HEAD_FORWARD_THRESHOLD = 0.08
MAX_IMAGE_DIM = 800
FRAME_SKIP = 5
YOLO_CONFIDENCE = 0.5

# ------------- MODELS -------------
@st.cache_resource
def load_models():
    engagement_model = joblib.load(r"C:\Users\EngagementDetector\engagement_detector.pkl")
    pose_detector = mp.solutions.pose.Pose(static_image_mode=True, min_detection_confidence=0.3)
    yolo_model = YOLO("yolov8n.pt")
    return engagement_model, pose_detector, yolo_model

model, pose_processor, yolo_model = load_models()

# ------------- FEATURES EXTRACTION -------------
def extract_enhanced_features(pose_data):
    pose_arr = np.array(pose_data)
    features = []

    left_hand_up = pose_arr[15,1] < pose_arr[13,1] + HAND_RAISE_THRESHOLD
    right_hand_up = pose_arr[16,1] < pose_arr[14,1] + HAND_RAISE_THRESHOLD
    head_forward = pose_arr[0,0] > pose_arr[11,0] + HEAD_FORWARD_THRESHOLD

    hands_distance = np.linalg.norm(pose_arr[15,:2] - pose_arr[16,:2])
    head_to_hands_avg = np.mean([
        np.linalg.norm(pose_arr[0,:2] - pose_arr[15,:2]),
        np.linalg.norm(pose_arr[0,:2] - pose_arr[16,:2])
    ])

    shoulder_angle = np.arctan2(
        pose_arr[12,1]-pose_arr[11,1],
        pose_arr[12,0]-pose_arr[11,0]
    )

    features.extend([left_hand_up, right_hand_up, head_forward, 
                     hands_distance, head_to_hands_avg, shoulder_angle])

    features.extend(pose_arr[:,:2].flatten())
    return features

# ------------- INFERENCE FUNCTIONS -------------
def detect_people(image_np):
    results = yolo_model.predict(image_np, classes=0, conf=YOLO_CONFIDENCE)
    boxes = results[0].boxes.xyxy.cpu().numpy() if results else []
    return boxes

def predict_engagement(person_img):
    pose_results = pose_processor.process(person_img)

    if pose_results.pose_landmarks:
        pose_data = [[lmk.x, lmk.y, lmk.visibility] for lmk in pose_results.pose_landmarks.landmark]
        features = extract_enhanced_features(pose_data)
        proba = model.predict_proba([features])[0][1]
        label = "Engaged ✅" if proba > 0.5 else "Disengaged ❌"
        return label, proba
    else:
        return None, None

def process_image(image_np):
    image_display = image_np.copy()
    people_boxes = detect_people(image_np)

    if len(people_boxes) == 0:
        st.error("❌ No people detected. Try another image.")
    else:
        for idx, box in enumerate(people_boxes):
            x1, y1, x2, y2 = map(int, box)
            person_crop = image_np[y1:y2, x1:x2]
            person_crop_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)

            label, proba = predict_engagement(person_crop_rgb)

            if label:
                cv2.rectangle(image_display, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(image_display, f"{label} ({proba:.2f})", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)

    st.image(image_display[..., ::-1], caption="Processed Image", use_column_width=True)

import pandas as pd

def process_video(video_file):
    temp_video_path = tempfile.NamedTemporaryFile(delete=False).name
    with open(temp_video_path, "wb") as f:
        f.write(video_file.read())

    cap = cv2.VideoCapture(temp_video_path)

    frame_count = 0
    engaged_counts = []
    frame_indices = []

    stframe = st.empty()
    chart_placeholder = st.empty()
    progress_bar = st.progress(0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % FRAME_SKIP == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            people_boxes = detect_people(frame_rgb)

            engaged_in_frame = 0
            total_people = 0

            for box in people_boxes:
                x1, y1, x2, y2 = map(int, box)
                person_crop = frame_rgb[y1:y2, x1:x2]

                label, proba = predict_engagement(person_crop)

                if label:
                    color = (0, 255, 0) if "Engaged" in label else (0, 0, 255)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, f"{label}", (x1, y1-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    if "Engaged" in label:
                        engaged_in_frame += 1
                    total_people += 1

            stframe.image(frame[..., ::-1], channels="RGB", use_column_width=True)

            if total_people > 0:
                engagement_pct = (engaged_in_frame / total_people) * 100
                engaged_counts.append(engagement_pct)
                frame_indices.append(frame_count)

                chart_data = pd.DataFrame({
                    'Frame': frame_indices,
                    'Engagement (%)': engaged_counts
                })
                chart_placeholder.line_chart(chart_data.set_index('Frame'))

        frame_count += 1
        progress_bar.progress(min(frame_count/500, 1.0))

    cap.release()
    os.unlink(temp_video_path)

    if engaged_counts:
        avg_engagement = np.mean(engaged_counts)
        st.success(f"✅ Average Engagement Across Video: {avg_engagement:.1f}%")
    else:
        st.warning("⚠️ No engagement detected.")

# ------------- STREAMLIT UI -------------


st.title("📚 Engagement Detection App")
st.markdown("Upload an **image** or **video** to predict engagement. Supports **multi-person detection**!")

tab1, tab2 = st.tabs(["📷 Image Upload", "🎥 Video Upload"])

with tab1:
    uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

    if uploaded_image:
        image = Image.open(uploaded_image)
        img_np = np.array(image)

        if img_np.shape[0] > MAX_IMAGE_DIM or img_np.shape[1] > MAX_IMAGE_DIM:
            img_np = cv2.resize(img_np, (MAX_IMAGE_DIM, MAX_IMAGE_DIM))

        process_image(img_np)

with tab2:
    uploaded_video = st.file_uploader("Upload a Video", type=["mp4", "avi", "mov"])

    if uploaded_video:
        process_video(uploaded_video)
