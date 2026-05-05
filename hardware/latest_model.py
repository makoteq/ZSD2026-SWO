import cv2
import torch
from ultralytics import YOLO
import tkinter as tk
from tkinter import filedialog
import sys
import os 
import psutil
import time
import numpy as np
import csv

def get_rpi_temperature():
    try:
        res = os.popen("vcgencmd measure_temp").readline()
        return float(res.replace("temp=", "").replace("'C\n", ""))
    except Exception:
        return 0.0 
        
model_path = "YOLOmodels/416_latest_full_integer_quant_edgetpu.tflite"
model = YOLO(model_path)

print("Model class name:", model.names)

NUMER_KLASY_AUTA = 0 
imgsz_val = 416

output_dir = "output"
os.makedirs(output_dir, exist_ok=True)
output_img = os.path.join(output_dir, "output_frame.jpg")
output_csv = os.path.join(output_dir, "metrics.csv")

root = tk.Tk()
root.withdraw()
video_path = filedialog.askopenfilename(
    title="Choose a video file",
    filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov"), ("All files", "*.*")]
)

if not video_path:
    print("No video file selected. Closing program.")
    sys.exit()

cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"ERROR, unable to open video: {video_path}")
    sys.exit()

unique_vehicle_ids = set()
metrics_data = []
frame_count = 0
warmup_frames = 30  
last_annotated_frame = None

print(f"Warmup: {warmup_frames}")

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("End of video or error reading frame.")
            break
            
        frame_count += 1
        t_start = time.time()
        
        results = model.track(
            source=frame, 
            conf=0.65,          
            iou=0.25,            
            tracker="bytetrack.yaml",       
            verbose=False, 
            classes=[NUMER_KLASY_AUTA], 
            imgsz=imgsz_val,
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
            
            if track_id != "N/A":
                unique_vehicle_ids.add(track_id)
                
            label_name = model.names[cls_id]     

            label_txt = f"id:{track_id} {label_name} {conf:.2f}"
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), thickness=2) 
            
            font_scale = 0.6  
            font_thickness = 1 
            (text_width, text_height), baseline = cv2.getTextSize(label_txt, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
            
            tlo_paska = (x1, y1 - text_height - 10)
            cv2.rectangle(annotated_frame, (x1, y1), (x1 + text_width, y1 - text_height - 10), (255, 0, 0), cv2.FILLED)
            
            cv2.putText(annotated_frame, label_txt, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness=font_thickness)
            
        last_annotated_frame = annotated_frame
        num_detections = len(boxes)
        avg_confidence = float(np.mean(boxes.conf.cpu().numpy())) if num_detections > 0 else 0.0
        
        t_end = time.time()
        real_fps = 1.0 / (t_end - t_start) if (t_end - t_start) > 0 else 0.0

        if frame_count > warmup_frames:
            speed = r.speed
            inference_ms = speed.get('inference', 0.0)

            metrics_data.append({
                'Model': os.path.basename(model_path),
                'Slice_Size': imgsz_val, 
                'Frame': frame_count - warmup_frames,
                'Total_FPS': round(real_fps, 2),
                'Inference_ms': round(inference_ms, 2),
                'Objects_On_Frame': num_detections,
                'Unique_Objects_Total': len(unique_vehicle_ids),
                'Avg_Confidence': round(avg_confidence, 3),
                'CPU_Percent': psutil.cpu_percent(),
                'RAM_Percent': psutil.virtual_memory().percent,
                'Temp_C': get_rpi_temperature()
            })   
            
        cv2.imshow("Vehicle Detection", annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Stopping video playback.")
            break

except KeyboardInterrupt:
    print("Playback interrupted by user.")

finally:
    cap.release()
    cv2.destroyAllWindows()
    
    if last_annotated_frame is not None:
        cv2.imwrite(output_img, last_annotated_frame)
        print(f"\nSaved frame to: {output_img}")

    if metrics_data:
        with open(output_csv, mode='w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=metrics_data[0].keys())
            writer.writeheader()
            writer.writerows(metrics_data)

        print(f"Real number of unique vehicles: {len(unique_vehicle_ids)}")
        print(f"Collected metrics (discarded {warmup_frames} warm-up frames) saved to: {output_csv}")