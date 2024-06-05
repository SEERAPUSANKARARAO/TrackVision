import cv2
import pandas as pd
from ultralytics import YOLO
from src.tracker import Tracker
import cvzone
from PIL import Image
import numpy as np
import streamlit as st
import os
import glob

# Load the YOLO model
model = YOLO('yolov8s.pt')

st.title("Object Recognition Dashboard")
st.sidebar.title("Upload a Video")
vid_bytes = st.sidebar.file_uploader("Upload a video", type=['mp4', 'mpv', 'avi'])

# Function to delete all files in the specified folder
def delete_old_videos(folder_path):
    files = glob.glob(os.path.join(folder_path, '*'))
    for f in files:
        os.remove(f)

# Directory where uploaded videos are saved
upload_folder = "uploaded_videos"

if not os.path.exists(upload_folder):
    os.makedirs(upload_folder)

if vid_bytes:
    # Clean up old videos
    delete_old_videos(upload_folder)

    vid_file = os.path.join(upload_folder, "uploaded." + vid_bytes.name.split('.')[-1])
    with open(vid_file, 'wb') as out:
        out.write(vid_bytes.read())
    
    # Open video capture
    cap = cv2.VideoCapture(vid_file)
else:
    st.warning("Please upload a video file.")
    st.stop()

width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Load class names
with open("coco.txt", "r") as my_file:
    class_list = my_file.read().strip().split("\n")

# Initialize counters and trackers
count = 0
persondown = {}
tracker = Tracker()
counter1 = []
personup = {}
counter2 = []
cy1 = 194  # y coordinate for the first line
cy2 = 220  # y coordinate for the second line
offset = 6
st.markdown("---")
output = st.empty()

def draw_boxes(img, results):
    detections = results[0].boxes.data.cpu().numpy()
    for detection in detections:
        x1, y1, x2, y2, score, class_id = map(int, detection[:6])
        class_name = class_list[class_id]
        if class_name == 'person':
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
    return img

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    count += 1
    if count % 3 != 0:
        continue

    frame = cv2.resize(frame, (1020, 500))

    # Run the model prediction
    results = model.predict(frame)

    list_detections = []
    for detection in results[0].boxes.data.cpu().numpy():
        x1, y1, x2, y2, score, class_id = map(int, detection[:6])
        class_name = class_list[class_id]
        if class_name == 'person':
            list_detections.append([x1, y1, x2, y2])

    # Update tracker
    bbox_id = tracker.update(list_detections)

    for bbox in bbox_id:
        x3, y3, x4, y4, id = bbox
        cx, cy = (x3 + x4) // 2, (y3 + y4) // 2
        cv2.circle(frame, (cx, cy), 4, (255, 0, 255), -1)

        # Check if person crosses the first line
        if cy1 - offset < cy < cy1 + offset:
            cvzone.putTextRect(frame, f'{id}', (x3, y3), scale=2, thickness=3)
            persondown[id] = (cx, cy)

        # Check if person crosses the second line
        if id in persondown:
            if cy2 - offset < cy < cy2 + offset:
                cvzone.putTextRect(frame, f'{id}', (x3, y3), scale=2, thickness=3)
                if id not in counter1:
                    counter1.append(id)

        # For counting people going up
        if cy2 - offset < cy < cy2 + offset:
            cvzone.putTextRect(frame, f'{id}', (x3, y3), scale=2, thickness=3)
            personup[id] = (cx, cy)

        if id in personup:
            if cy1 - offset < cy < cy1 + offset:
                cvzone.putTextRect(frame, f'{id}', (x3, y3), scale=2, thickness=3)
                if id not in counter2:
                    counter2.append(id)

    # Draw lines
    cv2.line(frame, (3, cy1), (1018, cy1), (0, 255, 0), 2)
    cv2.line(frame, (5, cy2), (1019, cy2), (0, 255, 255), 2)

    # Display counts
    down = len(counter1)
    up = len(counter2)
    cvzone.putTextRect(frame, f'Down: {down}', (50, 60), scale=2, thickness=2)
    cvzone.putTextRect(frame, f'Up: {up}', (50, 100), scale=2, thickness=2)

    # Draw detection boxes on the frame
    frame = draw_boxes(frame, results)

    # Convert frame to RGB format for Streamlit
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    output_img = Image.fromarray(frame_rgb)
    output.image(output_img)

cap.release()
cv2.destroyAllWindows()
