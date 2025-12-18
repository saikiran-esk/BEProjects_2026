"""
Face Recognition Streamlit App - Pavani Detection
Powered by Roboflow AI
"""

import streamlit as st
from pathlib import Path
import tempfile
import time
import os
import cv2
import numpy as np
from roboflow import Roboflow

# ---------------------------
# Page Config
# ---------------------------
st.set_page_config(
    page_title="Pavani Face Recognition", 
    layout="wide", 
    initial_sidebar_state="collapsed",
    page_icon="🎥"
)

# ---------------------------
# PRO UI/CSS
# ---------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

html, body, [class*="st-"], [class*="css-"] {
    font-family: 'Inter', sans-serif;
}

:root {
    --primary-color: #667eea;
    --secondary-color: #764ba2;
    --background-color: #0E1117;
    --secondary-background-color: #1A1E29;
    --text-color: #FAFAFA;
    --secondary-text-color: #ADB5BD;
}

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0E1117 0%, #1A1E29 100%);
    color: var(--text-color);
}

h1 {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    font-weight: 700;
    font-size: 2.8rem;
    padding-bottom: 0.5rem;
}

[data-testid="stButton"] button {
    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    color: #FFFFFF;
    border: none;
    border-radius: 8px;
    padding: 0.75rem 1.5rem;
    font-weight: 600;
    font-size: 1.1rem;
    width: 100%;
    transition: all 0.3s ease;
}

[data-testid="stButton"] button:hover {
    background: linear-gradient(135deg, #764ba2 0%, #667eea 100%);
    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
    transform: translateY(-2px);
}
</style>
""", unsafe_allow_html=True)

# ---------------------------
# Roboflow Initialization
# ---------------------------
@st.cache_resource
def init_roboflow():
    try:
        api_key = "lYQhNaqU50FdyzkR0Gq5"
        workspace = "project-nhn9q"
        project = "face-detection-w9fbh"
        version = 3
        person_name = "Pavani"
        
        rf = Roboflow(api_key=api_key)
        proj = rf.workspace(workspace).project(project)
        model = proj.version(version).model
        return model, workspace, person_name
    except Exception as e:
        st.error(f"❌ Failed to initialize Roboflow: {str(e)}")
        return None, None, None

# ---------------------------
# Video Processing Class (FIXED LOGIC)
# ---------------------------
class FaceDetectionProcessor:
    def __init__(self, model, person_name, confidence_threshold=0.4):
        self.model = model
        self.person_name = person_name
        self.confidence_threshold = confidence_threshold
    
    def process_video(self, input_path, output_path, progress_callback=None):
        cap = cv2.VideoCapture(str(input_path))
        if not cap.isOpened():
            raise RuntimeError("Failed to open video")
        
        fps = int(cap.get(cv2.CAP_PROP_FPS) or 30)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        
        # FIX: Try Browser Friendly Codec (H.264) first
        try:
            fourcc = cv2.VideoWriter_fourcc(*'avc1')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            if not out.isOpened(): raise Exception("avc1 failed")
        except:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        if not out.isOpened():
            raise RuntimeError("Failed to create output video")
        
        # FIX: Static file path to prevent AxiosError 400 (File Locking)
        temp_frame_path = "temp_inference_frame.jpg"
        
        frame_count = 0
        detection_count = 0
        total_confidence = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if progress_callback and total_frames > 0:
                progress_callback(frame_count / total_frames)
            
            # 1. Overwrite static file
            cv2.imwrite(temp_frame_path, frame)
            
            try:
                # 2. Predict using static path
                predictions = self.model.predict(
                    temp_frame_path,
                    confidence=int(self.confidence_threshold * 100)
                ).json()
                
                # 3. Draw
                for pred in predictions['predictions']:
                    x, y, w, h = pred['x'], pred['y'], pred['width'], pred['height']
                    conf = pred['confidence']
                    
                    x1, y1 = int(x - w/2), int(y - h/2)
                    x2, y2 = int(x + w/2), int(y + h/2)
                    
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
                    
                    label = f"{self.person_name} {conf*100:.1f}%"
                    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    
                    cv2.rectangle(frame, (x1, y1 - th - 10), (x1 + tw, y1), (0, 255, 0), -1)
                    cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                    
                    detection_count += 1
                    total_confidence += conf
                    
            except Exception:
                pass # Skip frame errors to prevent crash
            
            out.write(frame)
        
        cap.release()
        out.release()
        
        # Cleanup static file
        if os.path.exists(temp_frame_path):
            os.remove(temp_frame_path)
            
        stats = {
            'total_frames': frame_count,
            'detections': detection_count,
            'avg_confidence': (total_confidence / detection_count * 100) if detection_count > 0 else 0,
            'processing_time': time.time() - start_time,
            'fps': fps
        }
        return stats

# ---------------------------
# Main App UI
# ---------------------------
st.title("🎥 Face Recognition System")
st.markdown(f"**Detecting:** Pavani | **Powered by:** Roboflow AI")

with st.spinner("🔄 Initializing Roboflow model..."):
    model, workspace, person_name = init_roboflow()

if model is None:
    st.stop()

st.success(f"✅ Model loaded successfully! Detecting: **{person_name}**")

# Layout
st.markdown("### 1. Upload & Configure")
col1, col2 = st.columns([2, 1])

with col1:
    st.markdown("##### Upload Video")
    # FIX: Added 'key' to prevent upload reset
    uploaded_video = st.file_uploader(
        "Upload a video file",
        type=["mp4", "mov", "avi", "mkv"],
        label_visibility="collapsed",
        key="video_uploader_key"
    )
    
    if uploaded_video:
        st.video(uploaded_video)
        file_size = uploaded_video.size / (1024 * 1024)
        st.caption(f"📊 File: {uploaded_video.name} | Size: {file_size:.2f} MB")

with col2:
    st.markdown("##### Detection Settings")
    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.2, max_value=0.95, value=0.4, step=0.05
    ) 
    
    st.markdown("---")
    st.markdown("##### Model Info")
    st.info(f"**Person:** {person_name}\n**Workspace:** {workspace}")

st.markdown("---")

# Process Button
if st.button("🚀 Start Face Detection", use_container_width=True):
    if uploaded_video:
        with st.spinner("Processing video... This may take a few minutes."):
            tmp_dir = Path(tempfile.gettempdir()) / "face_recognition"
            tmp_dir.mkdir(exist_ok=True)
            
            in_path = tmp_dir / f"input_{int(time.time())}.mp4"
            
            # FIX: Seek(0) ensures clean read
            uploaded_video.seek(0)
            with open(in_path, "wb") as f:
                f.write(uploaded_video.read())
            
            out_path = tmp_dir / f"output_{int(time.time())}.mp4"
            
            progress_bar = st.progress(0.0)
            status_text = st.empty()
            
            def update_progress(fraction):
                progress_bar.progress(min(1.0, fraction))
                status_text.text(f"Processing: {fraction*100:.1f}%")
            
            try:
                processor = FaceDetectionProcessor(model, person_name, confidence_threshold)
                stats = processor.process_video(in_path, out_path, update_progress)
                
                progress_bar.progress(1.0)
                status_text.empty()
                st.toast("✅ Processing complete!", icon="🎉")
                
                # Results
                st.markdown("---")
                st.markdown("### 2. Results")
                
                res_col1, res_col2 = st.columns([2, 1])
                
                with res_col1:
                    if os.path.exists(out_path):
                        with open(out_path, "rb") as f:
                            video_bytes = f.read()
                        
                        st.video(video_bytes)
                        st.download_button(
                            "⬇️ Download Processed Video",
                            video_bytes,
                            file_name=f"{person_name}_detection.mp4",
                            mime="video/mp4",
                            use_container_width=True
                        )
                
                with res_col2:
                    st.metric("Face Detections", f"{stats['detections']}")
                    st.metric("Avg Confidence", f"{stats['avg_confidence']:.1f}%")
                    st.metric("FPS", f"{stats['fps']}")
                
                if os.path.exists(in_path): os.unlink(in_path)
                
            except Exception as e:
                st.error(f"❌ Processing failed: {str(e)}")
    else:
        st.warning("⚠️ Please upload a video first!")

st.markdown("---")
st.markdown("<p style='text-align: center; color: #666;'>Built with ❤️ using Streamlit & Roboflow</p>", unsafe_allow_html=True)
