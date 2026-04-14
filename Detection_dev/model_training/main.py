# Vehicle detection and classification
# ─────────────────────────────────────────────────────────────────────────────
# This script detects and tracks vehicles in a video using a custom YOLO model then classifies each vehicle type using a CNN model.
# Trajectories are drawn on the output video.
# Output notebook captures a crop of each detected vehicle, its type, confidence and k value, can be easily removed if not needed.
# Breaking distance is calculated based on k value from the CNN and a speed that has to be implemented.
#
# Setup:
#   python 3.12 required, because of tensorflow 2.16.1 compatibility.
#   pip install ultralytics==8.4.37 torch==2.11.0 torchvision==0.26.0 tensorflow==2.16.1 opencv-python==4.10.0.84 numpy==1.26.4 pillow==12.1.1 nbformat
#
# Folder structure:
#   content/
#     best.pt  -  custom YOLO detection model
#     carla_classification_cnn_model_v01.h5  -  CNN classification model
#     traffic.mp4  -  input video to process
#
# Output (written to content/):
#   trajectory.mp4  - annotated video with bounding boxes and trajectories
#   cars_output.ipynb - notebook with a crop, type, confidence and k per vehicle

import ultralytics
import tensorflow as tf
import torch
import cv2
import numpy as np
import nbformat
import base64
import io
from ultralytics import YOLO
from PIL import Image
from keras import models

print(f"Ultralytics version: {ultralytics.__version__}")
print(f"Torch version: {torch.__version__}")

# CONFIGURATION
VIDEO_PATH = "content/traffic.mp4"
START_TIME = 1
 
YOLO_MODEL_PATH = "content/best.pt"
CNN_MODEL_PATH = "content/carla_classification_cnn_model_v01.h5"
 
OUTPUT_VIDEO_PATH = "content/trajectory.mp4"
OUTPUT_NB_PATH = "content/cars_output.ipynb"
 
CONF_THRESHOLD = 0.5
IMGSZ = 800                     # 416 or 800 depends on best.pt model
ALLOWED_CLASSES_IDS = [0]
MAX_MISSING_FRAMES = float('inf')    # set to a number of frames to remove cars from carsDict if they are not seen for that many frames
 
VIDEO_FOURCC = 'mp4v'
TRACK_COLOR = (0, 255, 0)
BBOX_COLOR = (255, 255, 255)
LINE_THICKNESS = 1


class Car:
    CATEGORY_MAP = {
        0: "coupe",
        1: "hatchback",
        2: "sedan",
        3: "suv",
        4: "truck",
        5: "van",
    }
    IMG_SIZE = (128, 128)
 
    def __init__(self, trackId: int):
        self.trackId = trackId
        self.x = 0.0
        self.y = 0.0
        self.w = 0.0
        self.h = 0.0
        self.maxConfidence = 0.0
        self.last_confidence = 0.0
        self.history = []
        self.lastCrop = None
        self.last_seen = -1
        self.updateCount = 0
        self.type = "unknown"
        self.k = 0.0
        self.speed = 0.0
        self.breakingDistance = 0.0

 
    def checkType(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.IMG_SIZE)
        img_arr = np.array(img, dtype=np.float32) / 255.0
        img_batch = np.expand_dims(img_arr, axis=0)
 
        class_probs, k_pred = cnn.predict(img_batch, verbose=0)
 
        class_probs = class_probs[0]
        k_pred = float(k_pred[0][0])
        pred_idx = int(np.argmax(class_probs))
        confidence = float(class_probs[pred_idx])
        category = self.CATEGORY_MAP[pred_idx]
 
        return category, confidence, class_probs, k_pred
 
    def update(self, box: tuple, confidence: float, frame: np.ndarray, frame_index: int) -> None:
        self.updateCount += 1
        self.x, self.y, self.w, self.h = box
        self.last_confidence = confidence
        self.last_seen = frame_index
        self.history.append((float(self.x), float(self.y)))
 
        if confidence > self.maxConfidence:
            self.maxConfidence = confidence
 
            x1 = int(self.x - self.w / 2)
            y1 = int(self.y - self.h / 2)
            x2 = int(self.x + self.w / 2)
            y2 = int(self.y + self.h / 2)
 
            crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
            if crop.size > 0:
                self.lastCrop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                category, _, _, k = self.checkType(crop) # _,_ - previous confidence and probs
                self.type = category
                self.k = k

        # self.speed = ...
        self.breakingDistance = self.calcBreakingDistance()

 
    def calcBreakingDistance(self) -> float:
        return (self.speed ** 2) * self.k
 


def draw_custom_box(annotatedFrame, box_xyxy, trackId, conf, car_type, BOUNDING_BOX_COLOR, LINE_THICKNESS, font_scale=0.4, font_thickness=1):
    x1, y1, x2, y2 = map(int, box_xyxy)
    label_text = f"{car_type} {trackId} | {conf:.2f}"

    cv2.rectangle(annotatedFrame, (x1, y1), (x2, y2), BOUNDING_BOX_COLOR, LINE_THICKNESS)

    (text_width, text_height), _ = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
    cv2.rectangle(annotatedFrame, (x1, y1 - text_height - 5), (x1 + text_width, y1), BOUNDING_BOX_COLOR, -1)
    cv2.putText(annotatedFrame, label_text, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), font_thickness)


# this function is call in the end of main() to save the notebook with all captured vehicles, their ids, types, confidence and k values. 
def save_notebook(cars_dict, output_path):
    nb = nbformat.v4.new_notebook()
    cells = []
 
    for carId, carObj in cars_dict.items():
        if carObj.lastCrop is None:
            continue
 
        img_pil = Image.fromarray(carObj.lastCrop)
        h, w = carObj.lastCrop.shape[:2]
        buffer = io.BytesIO()
        img_pil.save(buffer, format="PNG")
        b64 = base64.b64encode(buffer.getvalue()).decode()
 
        caption = f"# Car ID: {carId} | Type: {carObj.type} | Conf: {carObj.maxConfidence:.2f} | k: {carObj.k:.4f}"
        cell = nbformat.v4.new_code_cell(source=caption)
        cell.outputs = [
            nbformat.v4.new_output(
                output_type="display_data",
                data={
                    "text/plain": caption,
                    "image/png": b64,
                },
                metadata={"image/png": {"width": w, "height": h}}
            )
        ]
        cells.append(cell)
 
    nb.cells = cells
    with open(output_path, "w", encoding="utf-8") as f:
        nbformat.write(nb, f)
    print(f"Captured {len(cells)} vehicles - {output_path}")



def main():
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    x = torch.rand(10000,10000).to(device)
    print(x.device)
 
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise IOError(f"Can't open {VIDEO_PATH}")
 
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        raise ValueError("Invalid frame rate in video file.")
 
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(START_TIME * fps))
 
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*VIDEO_FOURCC)
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, fps, (frameWidth, frameHeight))
    
    carsDict = {}
    frame_index = 0

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            results = model.track(
                source=frame,
                imgsz=IMGSZ, 
                conf=CONF_THRESHOLD,
                persist=True,
                verbose=False,
                device=0 if device == 'cuda' else 'cpu',
                tracker='bytetrack.yaml',
                classes=ALLOWED_CLASSES_IDS
            )


            annotatedFrame = frame.copy()

            if results[0].boxes.id is not None:
                boxes_xyxy = results[0].boxes.xyxy.cpu().numpy()
                boxes_xywh = results[0].boxes.xywh.cpu().numpy()

                trackIds = results[0].boxes.id.int().cpu().tolist()
                confidences = results[0].boxes.conf.cpu().tolist()

                for box_xyxy, box_xywh, trackId, conf in zip(boxes_xyxy, boxes_xywh, trackIds, confidences):

                    if trackId not in carsDict:
                        carsDict[trackId] = Car(trackId)

                    car = carsDict[trackId]
                    car.update(box_xywh, conf, frame, frame_index)

                    draw_custom_box(annotatedFrame, box_xyxy, trackId, conf, car.type, BBOX_COLOR, LINE_THICKNESS)

                    points = np.array(car.history).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotatedFrame, [points], isClosed=False, color=TRACK_COLOR, thickness=LINE_THICKNESS)

            stale_ids = [carId for carId, carObj in carsDict.items()
                        if carObj.last_seen < frame_index - MAX_MISSING_FRAMES
                        or carObj.last_confidence < CONF_THRESHOLD]
            for carId in stale_ids:
                del carsDict[carId]

            out.write(annotatedFrame)
            frame_index += 1

    except KeyboardInterrupt:
        pass

    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
    
    save_notebook(carsDict, OUTPUT_NB_PATH) # can be removed if you don't want to save the notebook with captured vehicles, their ids, types, confidence and k values.



model = YOLO(YOLO_MODEL_PATH)
cnn   = models.load_model(CNN_MODEL_PATH, compile=False)
 
if __name__ == "__main__":
    main()
