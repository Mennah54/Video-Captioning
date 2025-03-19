import os
import cv2
import numpy as np
from tqdm import tqdm

# مسارات البيانات
train_videos_path = r"C:\Mennah\semster 8\deep learning\assigments\video captioning\main_project\dataset\train"  # غير المسار لمسار مجلد التدريب
frames_output_path = "path/to/frames"  # مجلد تخزين الإطارات
os.makedirs(frames_output_path, exist_ok=True)

def extract_frames(video_path, output_path, frame_rate=5):
    """استخراج إطارات من فيديو بمعدل معين."""
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    save_folder = os.path.join(output_path, video_name)
    os.makedirs(save_folder, exist_ok=True)
    
    cap = cv2.VideoCapture(video_path)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    frame_interval = max(1, fps // frame_rate)
    
    frame_count = 0
    extracted_count = 0
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % frame_interval == 0:
            frame_path = os.path.join(save_folder, f"frame_{extracted_count}.jpg")
            cv2.imwrite(frame_path, frame)
            extracted_count += 1
        
        frame_count += 1
    
    cap.release()
    return extracted_count

# استخراج الإطارات لكل الفيديوهات في مجموعة التدريب
video_files = [f for f in os.listdir(train_videos_path) if f.endswith((".mp4", ".avi", ".mov", ".mkv"))]
for video in tqdm(video_files, desc="Extracting frames"):
    extract_frames(os.path.join(train_videos_path, video), frames_output_path)
