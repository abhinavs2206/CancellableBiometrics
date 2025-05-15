# import streamlit as st
# import cv2
# import numpy as np
# from backend import load_model, process_frame

# # Streamlit app
# st.title("Face Recognition System")

# # Model and classes
# MODEL_PATH = "model.pth"
# CLASSES = ['abhinav', 'andy', 'adarsh', 'adam', 'unknown_subjects']

# # Load model
# with st.spinner("Loading model..."):
#     load_model(MODEL_PATH)
# st.success("Model loaded successfully!")

# # Sidebar for video input
# st.sidebar.header("Video Input")
# input_option = st.sidebar.radio("Select input type:", ("Upload Video", "Webcam"))

# if input_option == "Upload Video":
#     uploaded_file = st.sidebar.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
#     if uploaded_file is not None:
#         # Save uploaded video to temporary file
#         with open("temp_video.mp4", "wb") as f:
#             f.write(uploaded_file.read())
#         video_path = "temp_video.mp4"
#     else:
#         video_path = None
# else:
#     video_path = 0  # Webcam

# # Main content
# if video_path is not None:
#     cap = cv2.VideoCapture(video_path)
#     stframe = st.empty()
#     info_container = st.sidebar.empty()
    
#     while cap.isOpened():
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         # Process frame
#         processed_frame, identity, confidence, box_size = process_frame(frame, CLASSES)
        
#         if processed_frame is not None:
#             # Convert frame to RGB for Streamlit
#             processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
#             stframe.image(processed_frame_rgb, use_column_width=True)
            
#             # Update sidebar information
#             if identity is not None:
#                 status = "Genuine" if identity != "unknown_subjects" else "Imposter"
#                 info_text = f"**Status**: {status}\n"
#                 info_text += f"**Identity**: {identity}\n"
#                 info_text += f"**Confidence**: {confidence:.2f}%\n"
#                 if box_size:
#                     info_text += f"**Bounding Box Size**: {box_size[0]:.2f}x{box_size[1]:.2f} pixels"
#                 info_container.markdown(info_text)
#             else:
#                 info_container.markdown("**Status**: No face detected")
        
#     cap.release()
# else:
#     st.warning("Please upload a video or select webcam to start.")



import streamlit as st
import cv2
import numpy as np
from backend import load_model, process_frame

# Streamlit app
st.title("Face Recognition System")

# Model and classes
MODEL_PATH = r"C:\Users\Abhinav Somisetty\Final_Project\model.pth"  # Update to your model.pth path if different
CLASSES = ['abhinav', 'andy', 'adarsh', 'adam', 'unknown_subjects']

# Load model
with st.spinner("Loading model..."):
    load_model(MODEL_PATH)
st.success("Model loaded successfully!")

# Sidebar for video input and information
st.sidebar.header("Video Input")
input_option = st.sidebar.radio("Select input type:", ("Upload Video", "Webcam"))

# Checkpoints
st.sidebar.header("Checkpoints")
checkpoint1_container = st.sidebar.empty()
checkpoint2_container = st.sidebar.empty()

# Confidence plot
st.sidebar.header("Confidence Over Time")
confidence_history = []
confidence_plot = st.sidebar.empty()

# Status information
st.sidebar.header("Recognition Status")
info_container = st.sidebar.empty()

# Checkpoint thresholds (based on bounding box width in pixels)
CHECKPOINT1_THRESHOLD = 100  # Pixels
CHECKPOINT2_THRESHOLD = 200  # Pixels

# CSS for checkpoint styling
st.markdown("""
    <style>
    .checkpoint {
        display: flex;
        align-items: center;
        margin-bottom: 10px;
    }
    .checkpoint-circle {
        width: 20px;
        height: 20px;
        border-radius: 50%;
        margin-right: 10px;
        display: inline-block;
    }
    .red { background-color: red; }
    .green { background-color: green; }
    </style>
""", unsafe_allow_html=True)

# Video input handling
if input_option == "Upload Video":
    uploaded_file = st.sidebar.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])
    if uploaded_file is not None:
        # Save uploaded video to temporary file
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.read())
        video_path = "temp_video.mp4"
    else:
        video_path = None
else:
    video_path = 0  # Webcam

# Main content
if video_path is not None:
    cap = cv2.VideoCapture(video_path)
    stframe = st.empty()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        # Process frame
        processed_frame, identity, confidence, box_size = process_frame(frame, CLASSES, confidence_threshold=0.7)
        
        if processed_frame is not None:
            # Convert frame to RGB for Streamlit
            processed_frame_rgb = cv2.cvtColor(processed_frame, cv2.COLOR_BGR2RGB)
            stframe.image(processed_frame_rgb, use_container_width=True)
            
            # Update confidence plot
            if confidence is not None:
                confidence_history.append(confidence)
                confidence_plot.line_chart(confidence_history)
            
            # Update checkpoints
            checkpoint1_status = "red"
            checkpoint2_status = "red"
            if box_size and box_size[0] >= CHECKPOINT1_THRESHOLD:
                checkpoint1_status = "green"
            if box_size and box_size[0] >= CHECKPOINT2_THRESHOLD:
                checkpoint2_status = "green"
            
            checkpoint1_container.markdown(
                f'<div class="checkpoint"><span class="checkpoint-circle {checkpoint1_status}"></span>Checkpoint 1</div>',
                unsafe_allow_html=True
            )
            checkpoint2_container.markdown(
                f'<div class="checkpoint"><span class="checkpoint-circle {checkpoint2_status}"></span>Checkpoint 2</div>',
                unsafe_allow_html=True
            )
            
            # Update sidebar information
            if identity is not None:
                status = "Genuine" if identity != "unknown_subjects" else "Imposter"
                info_text = f"**Status**: {status}\n"
                info_text += f"**Identity**: {identity}\n"
                info_text += f"**Confidence**: {confidence:.2f}%\n"
                if box_size:
                    info_text += f"**Bounding Box Size**: {box_size[0]:.2f}x{box_size[1]:.2f} pixels"
                info_container.markdown(info_text)
            else:
                info_container.markdown("**Status**: No face detected")
        
    cap.release()
else:
    st.warning("Please upload a video or select webcam to start.")