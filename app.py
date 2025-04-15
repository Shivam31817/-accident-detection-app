import streamlit as st
import tempfile
import cv2
from detector import detect_objects
from video_utils import extract_frames
from PIL import Image

st.set_page_config(page_title="🚦 Accident Detection", layout="centered")
st.title("🚗💥 Traffic Accident Detection from Video")
video_file = st.file_uploader("Upload dashcam/CCTV footage", type=["mp4", "mov", "avi"])

if video_file:
    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())
    st.video(tfile.name)

    if st.button("Run Accident Detection"):
        frames = extract_frames(tfile.name)

        stframe = st.empty()
        st.subheader("Analyzing...")
        accident_detected = False

        for i, frame in enumerate(frames[::10]):  # sample every 10th frame
            result = detect_objects(frame)
            if any(obj.cls == 2 or obj.cls == 5 for obj in result.boxes):  # cars, buses
                if len(result.boxes) > 6:  # crude logic: crowded frame
                    accident_detected = True
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            stframe.image(frame, caption=f"Frame {i}", use_column_width=True)

        if accident_detected:
            st.error("🚨 Accident Possibly Detected!")
        else:
            st.success("✅ No Accident Detected")
