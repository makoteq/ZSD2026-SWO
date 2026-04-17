import os
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from keras import models
from typing import Dict, Final
from pathlib import Path

# Importy lokalne (zakładam, że istnieją w Twoim projekcie)
from algorithms.lane_detection.lane_detector import LaneDetector
from utils.car import Car
from utils.utils import get_x_from_line

# Konfiguracja

CURRENT_SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(CURRENT_SCRIPT_PATH, "..", "data"))
VIDEO_PATH = os.path.join(DATA_DIR, "NormalTrafic3/rgb.mp4")
YOLO_MODEL_PATH = os.path.join(DATA_DIR, "models", "best.pt")
CNN_MODEL_PATH = os.path.join(DATA_DIR, "models", "cnn.h5")
OUTPUT_VIDEO_PATH = os.path.join(DATA_DIR, "output", "trajectory.mp4")

ROAD_WIDTH_METERS = 6.0
FOV = 50.0
START_TIME = 1
CONF_THRESHOLD = 0.5
IMGSZ = 800
ALLOWED_CLASSES_IDS = [0]
MAX_MISSING_FRAMES = 5
TRACK_COLOR = (0, 255, 0)
BBOX_COLOR = (255, 255, 255)
LINE_THICKNESS = 1
FONT_SCALE = 0.4
FONT_THICKNESS = 1
WINDOW_NAME = "Traffic Analysis"
WAIT_KEY_MS = 1
EXIT_KEY = ord('q')



def drawCustomBox(
    annotatedFrame: np.ndarray, 
    boxXyxy: np.ndarray, 
    trackId: int, 
    conf: float, 
    carType: str, 
    distance: float, 
    speed: float,
    realW: float,
    realH: float
) -> None:
    x1, y1, x2, y2 = map(int, boxXyxy)
    
    speedKmh = speed * 3.6
    line1 = f"{carType.upper()} ID:{trackId} | {conf:.2f}"
    line2 = f"D: {distance:.1f}m | S: {speedKmh:.1f}km/h"
    line3 = f"W: {realW:.2f}m | H: {realH:.2f}m"

    cv2.rectangle(annotatedFrame, (x1, y1), (x2, y2), BBOX_COLOR, LINE_THICKNESS)

    font = cv2.FONT_HERSHEY_SIMPLEX
    lineHeight = int(20 * FONT_SCALE + 10)
    bgWidth = 180
    bgHeight = lineHeight * 3 + 5

    cv2.rectangle(annotatedFrame, (x1, y1 - bgHeight), (x1 + bgWidth, y1), BBOX_COLOR, -1)

    cv2.putText(annotatedFrame, line1, (x1 + 5, y1 - (lineHeight * 2) - 5), font, FONT_SCALE, (0, 0, 0), FONT_THICKNESS)
    cv2.putText(annotatedFrame, line2, (x1 + 5, y1 - lineHeight - 5), font, FONT_SCALE, (0, 0, 0), FONT_THICKNESS)
    cv2.putText(annotatedFrame, line3, (x1 + 5, y1 - 5), font, FONT_SCALE, (255, 0, 0), FONT_THICKNESS)


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    model = YOLO(YOLO_MODEL_PATH)
    cnn = models.load_model(CNN_MODEL_PATH, compile=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_time = 1.0 / fps
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(START_TIME * fps))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, int(fps), (frameWidth, frameHeight))
    
    carsDict: Dict[int, Car] = {}
    frameIndex = 0
    


    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break
            
            if frameIndex == 0:
   
                detector = LaneDetector(str(script_dir / "algorithms" / "lane_detection" / "ROI_v2.pt"))
                detected_lines, y_horizon = detector.process_image(frame, debug=False)
                final_output = detector.draw_lanes(frame, detected_lines)
                print(f"Detected lines: {detected_lines}, Horizon Y: {y_horizon}")
                y=0
                xLeft = (detected_lines[0]['m'] * y) + detected_lines[0]['b']
                xRight = (detected_lines[1]['m'] * y) + detected_lines[1]['b']
                road_width_h0_px = abs(xRight - xLeft)
  

            results = model.track(
                source=frame, imgsz=IMGSZ, conf=CONF_THRESHOLD,
                persist=True, verbose=False, device=0 if device == 'cuda' else 'cpu',
                tracker='bytetrack.yaml', classes=ALLOWED_CLASSES_IDS
            ) 

            annotatedFrame = frame.copy()

            if results[0].boxes.id is not None:
                boxesXyxy = results[0].boxes.xyxy.cpu().numpy()
                boxesXywh = results[0].boxes.xywh.cpu().numpy()
                trackIds = results[0].boxes.id.int().cpu().tolist()
                confidences = results[0].boxes.conf.cpu().tolist()

                for boxXyxy, boxXywh, trackId, conf in zip(boxesXyxy, boxesXywh, trackIds, confidences):
                    if trackId not in carsDict:
                        carsDict[trackId] = Car(trackId)

                    car = carsDict[trackId]
              
                    car.update(
                    boxXywh,
                    conf,
                    frame,
                    frameIndex, 
                    cnn,
                    detected_lines,
                    road_width_h0_px,
                    ROAD_WIDTH_METERS,
                    FOV,
                    )

                    

                    drawCustomBox(annotatedFrame, boxXyxy, trackId, conf, car.type, car.distance[-1], car.velocity[-1], car.realWidth, car.realHeight)

                    points = np.array(car.history).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotatedFrame, [points], False, TRACK_COLOR, LINE_THICKNESS)

            staleIds = [carId for carId, carObj in carsDict.items() if carObj.lastSeen < frameIndex - 5]
            for carId in staleIds: del carsDict[carId]

            out.write(annotatedFrame)
            cv2.imshow(WINDOW_NAME, annotatedFrame)
            if cv2.waitKey(WAIT_KEY_MS) & 0xFF == EXIT_KEY: break
            frameIndex += 1

    except Exception as e:
        print(f"Błąd: {e}")
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()