import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import pandas as pd
import cv2
import os
from PIL import Image

# Load action labels
action_labels = pd.read_csv("data/Training_set.csv").label

# Custom CSS for styling with animations
st.markdown(
    """
    <style>
    body {
        background-color: #f8f9fa;
        font-family: 'Helvetica Neue', Arial, sans-serif;
    }
    .main-title {
        font-size: 42px;
        color: #1E3A8A;
        font-weight: 700;
        text-align: center;
        margin-bottom: 20px;
        text-shadow: 0px 1px 2px rgba(0, 0, 0, 0.1);
    }
    .description {
        font-size: 20px;
        color: #4B5563;
        text-align: center;
        margin-bottom: 30px;
        font-weight: 400;
        line-height: 1.5;
    }
    .subheader {
        font-size: 28px;
        color: #2563EB;
        margin-top: 20px;
        margin-bottom: 15px;
        text-align: center;
        font-weight: 600;
    }
    .stButton>button {
        background-color: #2563EB !important;
        color: white !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-size: 16px !important;
        font-weight: 500 !important;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1) !important;
        transition: all 0.2s ease !important;
    }
    .stButton>button:hover {
        background-color: #1D4ED8 !important;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1) !important;
    }
    .stSpinner {
        color: #2563EB;
    }
    .success-message {
        background-color: #ECFDF5;
        color: #065F46;
        padding: 16px;
        border-radius: 8px;
        font-weight: 500;
        margin-top: 20px;
        border-left: 4px solid #10B981;
    }
    .upload-section {
        background-color: #F3F4F6;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    .stTabs [data-baseweb="tab"] {
        background-color: #E5E7EB;
        border-radius: 8px 8px 0px 0px;
        padding: 10px 16px;
        font-weight: 500;
    }
    .stTabs [aria-selected="true"] {
        background-color: #2563EB !important;
        color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Title and description
st.markdown('<div class="main-title">🌟 Motion Sensing System 🌟</div>', unsafe_allow_html=True)
st.markdown('<div class="description">Upload a video or image to predict the action being performed with our advanced AI model</div>', unsafe_allow_html=True)

# Tabs for Video and Image
tab1, tab2 = st.tabs(["🎥 Video Analysis", "🖼️ Image Analysis"])

# Helper function to process video frames
def process_video(video_path, input_size=(100, 100)):
    frames = []
    cap = cv2.VideoCapture(video_path)
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, input_size)
        frame = frame / 255.0  # Normalize to [0, 1]
        frames.append(frame)
    cap.release()
    return np.array(frames)

# Model loading
@st.cache_resource
def load_cnn_model(path="model/motion_sensing_system.h5"):
    return load_model(path)

# Video-based prediction tab
with tab1:
    st.markdown('<div class="subheader">🎥 Video Prediction</div>', unsafe_allow_html=True)
    video_file = st.file_uploader("📂 Upload a video file (mp4, avi, mov)", type=["mp4", "avi", "mov"])

    if video_file:
        # Save the uploaded video temporarily
        temp_video_path = "temp_video.mp4"
        with open(temp_video_path, "wb") as f:
            f.write(video_file.read())

        # Add progress bar for processing
        st.write("🔄 Processing video...")
        progress_bar = st.progress(0)

        # Process video frames
        frames = process_video(temp_video_path)
        for i in range(100):
            progress_bar.progress(i + 1)

        # Load model and make predictions
        with st.spinner("🔍 Predicting..."):
            model = load_cnn_model()
            predictions = model.predict(frames)

        # Aggregate predictions (e.g., majority vote, average score)
        action_index = np.argmax(np.mean(predictions, axis=0))
        predicted_action = action_labels[action_index]

        # Display results with animation
        st.balloons()
        st.success(f"✅ Predicted Action: **{predicted_action}** 🎉")

        # Clean up the temporary video
        os.remove(temp_video_path)

# Image-based prediction tab
with tab2:
    st.markdown('<div class="subheader">🖼️ Image Prediction</div>', unsafe_allow_html=True)
    image_file = st.file_uploader("📂 Upload an image file (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

    if image_file:
        # Save the uploaded image temporarily
        temp_image_path = "temp_image.jpg"
        with open(temp_image_path, "wb") as f:
            f.write(image_file.read())

        # Display the uploaded image
        st.image(temp_image_path, caption="📸 Uploaded Image", use_column_width=True)

        # Preprocess the image
        def preprocess_image(image_path, input_size=(100, 100)):
            image = Image.open(image_path).convert("RGB")
            image = image.resize(input_size)
            image_array = np.array(image) / 255.0  # Normalize to [0, 1]
            return np.expand_dims(image_array, axis=0)  # Add batch dimension

        # Predict the action
        with st.spinner("🔍 Predicting..."):
            model = load_cnn_model()
            processed_image = preprocess_image(temp_image_path)
            prediction = model.predict(processed_image)

            # Get the action label
            action_index = np.argmax(prediction[0])
            predicted_action = action_labels[action_index]

        # Display the result with animation
        st.balloons()
        st.success(f"✅ Predicted Action: **{predicted_action}** 🎉")

        # Clean up the temporary image
        os.remove(temp_image_path)
