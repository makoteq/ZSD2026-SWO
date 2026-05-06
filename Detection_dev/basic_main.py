import cv2
import torch
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog
import sys

model_path = "Detection_dev/best.pt"
model = YOLO(model_path)

print("Model class name:", model.names)

NUMER_KLASY_AUTA = 0 

root = tk.Tk()
root.withdraw()
video_path = filedialog.askopenfilename(
    title="Choose a video file",
    filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov"), ("All files", "*.*")]
)

# Security check for video file selection
if not video_path:
    print("No video file selected. Closing program.")
    sys.exit()

cap = cv2.VideoCapture(video_path)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("End of video or error reading frame.")
            break

        # Switch from predict to track and added persist
        results = model.track(
            source=frame, 
            conf=0.5,       
            verbose=False, 
            classes=[NUMER_KLASY_AUTA], 
            device=0 if device == 'cuda' else 'cpu',
            imgsz=416,
            persist=True
        )

        annotated_frame = frame.copy()
        r = results[0]
        boxes = r.boxes 
        
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0]) 
            conf = float(box.conf[0])            
            cls_id = int(box.cls[0])             
            track_id = int(box.id[0]) if box.id is not None else "N/A" 
            label_name = model.names[cls_id]     

            label_txt = f"id:{track_id} {label_name} {conf:.2f}"
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), thickness=2) 
            
            font_scale = 0.6  
            font_thickness = 1 
            (text_width, text_height), baseline = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            
            tlo_paska = (x1, y1 - text_height - 10)
            cv2.rectangle(annotated_frame, (x1, y1), (x1 + text_width, y1 - text_height - 10), (255, 0, 0), cv2.FILLED)
            
            cv2.putText(annotated_frame, label_txt, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness=font_thickness)
            
        # play the annotated frame in a window
        cv2.imshow("Vehicle Detection", annotated_frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Stopping video playback.")
            break

except KeyboardInterrupt:
    print("Playback interrupted by user.")

finally:
    cap.release()
    cv2.destroyAllWindows()