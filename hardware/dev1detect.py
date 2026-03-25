import cv2
import torch
import numpy as np
import ultralytics
from collections import defaultdict
from ultralytics import YOLO
 
print(f"Ultralytics version: {ultralytics.__version__}")
print(f"Torch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
 
 
print(f"Using device: {device}")
 
model = YOLO('240_yolov8n_full_integer_quant_edgetpu.tflite')

video_path = r"traffic.mp4"
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    print(f"cant load file {video_path}")
    exit()
 
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out = cv2.VideoWriter('trajectory.mp4', fourcc, fps, (frame_width, frame_height))

track_history = defaultdict(lambda: [])

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
 
        results = model.track(
            imgsz=1280,
            source=frame, 
            conf=0.1, 
            persist=True,  
            verbose=False, 
            device="tpu:0"
        )
 
        annotated_frame = results[0].plot()

        if results[0].boxes.id is not None:
            boxes = results[0].boxes.xywh.cpu()
            track_ids = results[0].boxes.id.int().cpu().tolist()
 
            for box, track_id in zip(boxes, track_ids):
                x, y, w, h = box
                track = track_history[track_id]
                
                track.append((float(x), float(y + h/2)))  
                
                points = np.array(track).astype(np.int32).reshape((-1, 1, 2))
                cv2.polylines(annotated_frame, [points], isClosed=False, color=(0, 255, 0), thickness=2)
        annotated_frame = cv2.resize(annotated_frame, (frame_width, frame_height))
        out.write(annotated_frame)
        print(".", end="", flush=True)

        cv2.imshow("YOLO Tracking", annotated_frame)
 
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("stop")
            break
 
except KeyboardInterrupt:
    print("manually stopped")
 
finally:
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    print("results in trajectory.mp4")
