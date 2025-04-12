import cv2 as cv
import os
import shutil


def extract_frames(video_path, output_folder, frame_name):

    os.makedirs(output_folder, exist_ok=True)

    cap = cv.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break 
        
        if frame_count%2 == 0:
            frame_filename = os.path.join(output_folder, f"{frame_name}{frame_count:05d}.jpg")
            cv.imwrite(frame_filename, frame)
            print(f"Saved {frame_filename}")
        frame_count += 1

    cap.release()
    print("Frame extraction completed.")

def move_images(source_folder, destination_folder, st, end):
    os.makedirs(destination_folder, exist_ok=True)

    for filename in os.listdir(source_folder):
        num = int(filename[10:15])
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.gif')) and st<=num<=end:
            src_path = os.path.join(source_folder, filename)
            dst_path = os.path.join(destination_folder, filename)
            shutil.move(src_path, dst_path)
            print(f"Moved {filename} to {destination_folder}")
