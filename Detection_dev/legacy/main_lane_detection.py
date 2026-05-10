import cv2
import torch
import sys
import tkinter as tk
from tkinter import filedialog
from ultralytics import YOLO

from algorithms.lane_detection.lane_detector import LaneDetector

MODEL_PATH = "../data/models/best.pt"
VEHICLE_CLASS_ID = 0
TARGET_WIDTH = 400
CONFIDENCE_THRESHOLD = 0.5
IMAGE_SIZE = 416

# Load YOLO model
model = YOLO(MODEL_PATH)
print("Model class names:", model.names)

root = tk.Tk()
root.withdraw()
root.attributes('-topmost', True)

video_file_path = filedialog.askopenfilename(
    title="Choose a video file",
    filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov"), ("All files", "*.*")]
)
root.destroy()

# Security check for video file selection
if not video_file_path:
    print("No video file selected. Exiting program.")
    sys.exit()

cap = cv2.VideoCapture(video_file_path)
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Initialize Lane Detector
lane_detector = LaneDetector()
detected_lines = []
is_first_frame = True


# MAIN VIDEO PROCESSING LOOP

try:
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            print("End of video or error reading frame.")
            break

        # Detect lanes ONLY on the first frame
        if is_first_frame:
            detected_lines = lane_detector.detect(frame)
            is_first_frame = False

        # Track vehicles using YOLO
        results = model.track(
            source=frame,
            conf=CONFIDENCE_THRESHOLD,
            verbose=False,
            classes=[VEHICLE_CLASS_ID],
            device=0 if device == 'cuda' else 'cpu',
            imgsz=IMAGE_SIZE,
            persist=True
        )

        annotated_frame = frame.copy()

        # Draw lane lines
        if detected_lines:
            annotated_frame = lane_detector.draw_lines(annotated_frame, detected_lines)

        current_result = results[0]

        for box in current_result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            confidence = float(box.conf[0])
            class_id = int(box.cls[0])
            track_id = int(box.id[0]) if box.id is not None else "N/A"
            label_name = model.names[class_id]

            label_text = f"ID:{track_id} {label_name} {confidence:.2f}"

            # Draw vehicle bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), thickness=2)

            # Text properties
            font_scale = 0.6
            font_thickness = 1
            (text_width, text_height), baseline = cv2.getTextSize(
                label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness
            )

            # Draw solid background rectangle for text
            cv2.rectangle(
                annotated_frame,
                (x1, y1),
                (x1 + text_width, y1 - text_height - 10),
                (255, 0, 0),
                cv2.FILLED
            )

            # Draw text over the background
            cv2.putText(
                annotated_frame,
                label_text,
                (x1, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness=font_thickness
            )

        # Resize frame for screen display
        original_height, original_width = annotated_frame.shape[:2]
        scale_ratio = TARGET_WIDTH / original_width

        new_width = int(original_width * scale_ratio)
        new_height = int(original_height * scale_ratio)

        resized_frame = cv2.resize(annotated_frame, (new_width, new_height))

        cv2.imshow("Vehicle Detection", resized_frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            print("Stopping video playback.")
            break

except KeyboardInterrupt:
    print("Playback interrupted by user.")

finally:
    cap.release()
    cv2.destroyAllWindows()