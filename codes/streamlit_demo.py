import sys
sys.path.append('/Users/saisrikanth/Desktop/suchama assignment')

import os
import torch
import streamlit as st

torch.classes.__path__ = []

import tempfile
import cv2
from pathlib import Path
import uuid
from codes.YOLO import YOLO_Inference
from ultralytics import YOLO
import time as pytime

st.title("📹 Video Upload & Processing App")

sop = ['Remove the camera protective film',
       'Scan the display side of the phone',
       'Place the display side with the battery on the belt',
       'Remove the film from the other side',
       'Place the back side on another belt']

# Initialize session state
if 'sop_initialized' not in st.session_state:
    st.session_state.sop_initialized = True
    st.session_state.number_of_cycles = 0
    st.session_state.number_of_deviations = 0
    st.session_state.current_action_idx = 0
    st.session_state.time_taken = "0s"
# if 'total_time_taken' not in st.session_state:
    st.session_state.total_time_taken = 0.0
    st.session_state.completed_steps = [False] * len(sop)

# Sidebar display
st.sidebar.title("Process Info")
st.sidebar.subheader("Standard Operating Procedure (SOP)")
print('---------------')
print(st.session_state.completed_steps)
for i, step in enumerate(sop):
    if st.session_state.completed_steps[i]:
        st.sidebar.markdown(f"✅ <span style='color:green'>{step}</span>", unsafe_allow_html=True)
    else:
        st.sidebar.markdown(f"🔲 {step}")

number_of_cycles_placeholder = st.sidebar.empty()
time_taken_placeholder = st.sidebar.empty()
total_time_taken_placeholder = st.sidebar.empty()

number_of_cycles_placeholder.subheader("Number of Cycles")
time_taken_placeholder.subheader("Time Taken")
total_time_taken_placeholder.subheader("Total Time Taken")

# 1. Upload video
uploaded_file = st.file_uploader("Upload a video file", type=["mp4", "avi", "mov"])

if uploaded_file is not None:
    # Save uploaded file to temp
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
    temp_input.write(uploaded_file.read())
    temp_input.flush()

    output_path = os.path.join(tempfile.gettempdir(), f"{uuid.uuid4().hex}_processed.mp4")

    # Load YOLO model
    model = YOLO('/Users/saisrikanth/Desktop/suchama assignment/best (4).pt')
    st.success("✅ Model loaded")

    # Prepare video
    cap = cv2.VideoCapture(temp_input.name)
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'avc1')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    stframe = st.empty()
    deviation_alert = st.empty()
    status_placeholder = st.empty()

    start_time = None
    frame_count = 0

    st.info("🔄 Processing video... Please wait...")

    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        T = frame_count / fps

        results = model.predict(frame, verbose=False)
        pred = YOLO_Inference.get_class_conf(results, T)

        if pred:
            pred_idx = sop.index(pred[0])
            if pred_idx == st.session_state.current_action_idx:
                st.session_state.completed_steps[pred_idx] = True
                if pred_idx == 0:
                    start_time = pred[2]
                st.session_state.current_action_idx += 1
            elif pred_idx != st.session_state.current_action_idx - 1:
                st.session_state.number_of_deviations += 1
                print(f"Deviation at {pred[2]}")
                deviation_alert.warning(f"⚠️ Deviation detected at {round(pred[2], 2)}s — Expected step: '{sop[st.session_state.current_action_idx]}', but got: '{pred[0]}'")


            if st.session_state.current_action_idx == len(sop):
                st.session_state.number_of_cycles += 1
                st.session_state.current_action_idx = 0
                end_time = pred[2]
                cycle_time = round(end_time - start_time, 2)
                st.session_state.time_taken = f"{cycle_time}s"
                st.session_state.total_time_taken += cycle_time
                st.session_state.completed_steps = [False] * len(sop)


        # Update sidebar live
        number_of_cycles_placeholder.markdown(f"### Number of Cycles\n**{st.session_state.number_of_cycles}**")
        time_taken_placeholder.markdown(f"### Time Taken\n**{st.session_state.time_taken}**")
        total_time_taken_placeholder.markdown(f"### Total Time Taken\n**{round(st.session_state.total_time_taken, 2)}s**")

        # Annotate and show frame
        annotated_frame = results[0].plot()
        annotated_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        stframe.image(annotated_frame, channels="RGB", caption=f"Frame {frame_count}")

        # Save to output video
        out.write(cv2.cvtColor(annotated_frame, cv2.COLOR_RGB2BGR))

        frame_count += 1

    cap.release()
    out.release()

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        st.success("✅ Video processed successfully!")
    #     st.video(output_path)
    # else:
    #     st.error("❌ Inference failed or produced empty output.")
