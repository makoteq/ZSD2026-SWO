import os
import time
import ultralytics
import tensorflow as tf
import torch
import cv2
import numpy as np
import nbformat
import base64
import io
import timm
from ultralytics import YOLO
from PIL import Image
from keras import models
from typing import Dict, Final
import matplotlib.pyplot as plt
from pathlib import Path
from algorithms.lane_detection.lane_detector import LaneDetector

from utils.radar import Radar
from utils.car import Car
from utils.depth import Depth

CURRENT_SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(CURRENT_SCRIPT_PATH, "..", "data"))
VIDEO_PATH = os.path.join(DATA_DIR, "normal_traffic\\rgb.mp4")
YOLO_MODEL_PATH = os.path.join(DATA_DIR, "models", "best.pt")
CNN_MODEL_PATH = os.path.join(DATA_DIR, "models", "cnn.h5")
DEPTH_MODEL_PATH = os.path.join(DATA_DIR, "models")
OUTPUT_VIDEO_PATH = os.path.join(DATA_DIR, "output", "trajectory.mp4")
OUTPUT_NB_PATH = os.path.join(DATA_DIR, "output", "cars_output.ipynb")
START_TIME = 1
CONF_THRESHOLD = 0.5
IMGSZ = 800
ALLOWED_CLASSES_IDS = [0]
MAX_MISSING_FRAMES = float('inf')
VIDEO_FOURCC = 'mp4v'
TRACK_COLOR = (0, 255, 0)
BBOX_COLOR = (255, 255, 255)
LINE_THICKNESS = 1
FONT_SCALE = 0.4
FONT_THICKNESS = 1
WINDOW_NAME = "Traffic Analysis"
WAIT_KEY_MS = 1
EXIT_KEY = ord('q')
DRAW_LINES = 1

DEPTH_CMAP: Final[str] = "magma"
DEPTH_TITLE: Final[str] = "MiDaS Depth Map"
COLORBAR_LABEL: Final[str] = "Relative Depth Intensity"
FIG_SIZE_DEPTH: Final[tuple[int, int]] = (10, 6)

def activateAlarm():
    print("ALARM!!!")


def isPointOutsideDetectedLanes(pointX: float, pointY: float, detectedLines: list, frameWidth: int, yHorizon: int) -> bool:
    if len(detectedLines) < 2:
        return False

    if pointY < yHorizon:
        return False

    laneXs = [float(lane["m"] * pointY + lane["b"]) for lane in detectedLines]
    leftBoundary = max(0.0, min(laneXs))
    rightBoundary = min(float(frameWidth - 1), max(laneXs))

    return pointX < leftBoundary or pointX > rightBoundary


def drawCustomBox(annotatedFrame: np.ndarray, boxXyxy: np.ndarray, trackId: int, conf: float, carType: str) -> None:
    x1, y1, x2, y2 = map(int, boxXyxy)
    labelText = f"{carType} {trackId} | {conf:.2f}"

    cv2.rectangle(annotatedFrame, (x1, y1), (x2, y2), BBOX_COLOR, LINE_THICKNESS)

    (textWidth, textHeight), _ = cv2.getTextSize(labelText, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS)
    cv2.rectangle(annotatedFrame, (x1, y1 - textHeight - 5), (x1 + textWidth, y1), BBOX_COLOR, -1)
    cv2.putText(annotatedFrame, labelText, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 0), FONT_THICKNESS)


if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent


    model = YOLO(YOLO_MODEL_PATH)
    cnn = models.load_model(CNN_MODEL_PATH, compile=False)
    detector = LaneDetector(str(script_dir / "algorithms" / "lane_detection" / "ROI_v2.pt"))
    MODEL_PATH = "data/models/depth.pth"
    LIB_PATH = "data/models/Depth-Anything-V2" 

    # depthProcessor = Depth(modelPath=MODEL_PATH, libPath=LIB_PATH)
    # depthProcessor.downloadModel()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        exit(1)

    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(START_TIME * fps))

    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*VIDEO_FOURCC)
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, int(fps), (frameWidth, frameHeight))
    
    carsDict: Dict[int, Car] = {}
    detected_lines = []
    y_horizon = 0
    frameIndex = 0

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            if frameIndex == 0:

                #depth estimation 1
                # depthMap = depthProcessor.getDepthMap(frame)
                # depthProcessor.getLog()

                # lane detection
                detected_lines, y_horizon = detector.process_image(frame, debug=False)
                print('detected_lines')
                print(detected_lines)
                final_output = detector.draw_lanes(frame, detected_lines)
                target_width = 800
                aspect_ratio = target_width / final_output.shape[1]
                target_height = int(final_output.shape[0] * aspect_ratio)
                resized_output = cv2.resize(final_output, (target_width, target_height))
                cv2.imshow("Final Result", resized_output)
            if DRAW_LINES:
                frame = detector.draw_lanes(frame, detected_lines)

            #depth estimation 2
            #     # Wizualizacja
            #     vis = depthProcessor.visualizeDepth(depthMap)
            #     cv2.imshow("Depth Anything V2 Small", vis)
            #     cv2.waitKey(0)


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
                boxesXyxy = results[0].boxes.xyxy.cpu().numpy()
                boxesXywh = results[0].boxes.xywh.cpu().numpy()
                trackIds = results[0].boxes.id.int().cpu().tolist()
                confidences = results[0].boxes.conf.cpu().tolist()

                for boxXyxy, boxXywh, trackId, conf in zip(boxesXyxy, boxesXywh, trackIds, confidences):
                    if trackId not in carsDict:
                        carsDict[trackId] = Car(trackId)

                    car = carsDict[trackId]
                    car.update(boxXywh, conf, frame, frameIndex, cnn)

                    drawCustomBox(annotatedFrame, boxXyxy, trackId, conf, car.type)

                    points = np.array(car.history).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotatedFrame, [points], isClosed=False, color=TRACK_COLOR, thickness=LINE_THICKNESS)

                    if car.history:
                        pointX, pointY = car.history[-1]
                        if isPointOutsideDetectedLanes(pointX, pointY, detected_lines, frameWidth, y_horizon):
                            activateAlarm()

            staleIds = [carId for carId, carObj in carsDict.items()
                        if carObj.lastSeen < frameIndex - MAX_MISSING_FRAMES
                        or carObj.lastConfidence < CONF_THRESHOLD]
            for carId in staleIds:
                del carsDict[carId]

            out.write(annotatedFrame)
            cv2.imshow(WINDOW_NAME, annotatedFrame)
            
            if cv2.waitKey(WAIT_KEY_MS) & 0xFF == EXIT_KEY:
                break
                
            frameIndex += 1

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()