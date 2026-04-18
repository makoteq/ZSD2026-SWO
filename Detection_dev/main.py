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
from utils.radar import SENSOR_PITCH_DEG, SENSOR_YAW_DEG, Radar
from utils.utils import get_x_from_line, drawCustomBox


CURRENT_SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(CURRENT_SCRIPT_PATH, "..", "data"))
VIDEO_PATH = os.path.join(DATA_DIR, "normalTraffic_DistMarkers/rgb.mp4")
CSV_PATH = os.path.join(DATA_DIR, "normalTraffic_DistMarkers/radar_points_world.csv")
YOLO_MODEL_PATH = os.path.join(DATA_DIR, "models", "best.pt")
CNN_MODEL_PATH = os.path.join(DATA_DIR, "models", "cnn.h5")
OUTPUT_VIDEO_PATH = os.path.join(DATA_DIR, "output", "trajectory.mp4")

# yolo
ROAD_WIDTH_METERS = 7.0
FOV = 14
START_TIME = 10
CONF_THRESHOLD = 0.8
IMGSZ = 800
ALLOWED_CLASSES_IDS = [0]
MAX_MISSING_FRAMES = 5
LINE_THICKNESS = 1
TRACK_COLOR = (0, 255, 0)

TEXT_COLOR: Final[tuple] = (255, 255, 255)
TEXT_THICKNESS: Final[int] = 2
TEXT_SCALE: Final[float] = 0.7
TEXT_POSITION_X: Final[int] = 20
TEXT_POSITION_Y_START: Final[int] = 30
TEXT_LINE_SPACING: Final[int] = 30

# radar
RADAR_STEP_INTERVAL = 10
SENSOR_PITCH_DEG = 0.0
SENSOR_YAW_DEG = 0.0
SENSOR_ROLL_DEG = 0.0
CAMERA_HEIGHT_OFFSET = 6.0
MASK_Z_MIN = 30.0
MASK_Z_MAX = 50.0
MASK_Y_MIN =75.0
MASK_Y_MAX = 130.0

# window
WINDOW_NAME = "Traffic Analysis"
WAIT_KEY_MS = 1
EXIT_KEY = ord('q')

if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    model = YOLO(YOLO_MODEL_PATH)
    cnn = models.load_model(CNN_MODEL_PATH, compile=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        exit(1)

    radar: Radar = Radar(CSV_PATH, START_TIME)
    radar.adjustPoints(SENSOR_PITCH_DEG, SENSOR_YAW_DEG, SENSOR_ROLL_DEG, CAMERA_HEIGHT_OFFSET)
    radar.applyMask(MASK_Z_MIN, MASK_Z_MAX, MASK_Y_MIN, MASK_Y_MAX)


    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"FPS: {fps}")
    frame_time = 1.0 / fps
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(START_TIME * fps))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, int(fps), (frameWidth, frameHeight))
    
    carsDict: Dict[int, Car] = {}
    frameIndex = 0
    roadWidthH0Px = 0.0
    


    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break

           
            if frameIndex == 0:
                detector = LaneDetector(str(script_dir / "algorithms" / "lane_detection" / "ROI_v2.pt"))
                detected_lines, y_horizon = detector.process_image(frame, debug=False)
                
                # Filler / Lorem Ipsum dla linii, jeśli ich nie wykryto
                dummyLine = {'m': 0.0, 'b': 0.0, 'x_bot': 0.0, 'abs_m': 0.0}
                while len(detected_lines) < 2:
                    detected_lines.append(dummyLine)

                y = frameHeight
                xLeft = (detected_lines[0]['m'] * y) + detected_lines[0]['b']
                xRight = (detected_lines[1]['m'] * y) + detected_lines[1]['b']
                road_width_h0_px = abs(xRight - xLeft) if abs(xRight - xLeft) != 0 else float(frameWidth)

  
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
                        frame_time,
                        IMGSZ
                    )

                    drawCustomBox(annotatedFrame, boxXyxy, trackId, conf, car.type, car.distance[-1], car.velocity[-1], car.realWidth, car.realHeight)

                    points = np.array(car.history).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotatedFrame, [points], False, TRACK_COLOR, LINE_THICKNESS)

            currentTime = START_TIME + (frameIndex * frame_time)
            cv2.putText(annotatedFrame, f"Frame: {frameIndex}", (TEXT_POSITION_X, TEXT_POSITION_Y_START), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS)
            cv2.putText(annotatedFrame, f"Time: {currentTime:.2f}s", (TEXT_POSITION_X, TEXT_POSITION_Y_START + TEXT_LINE_SPACING), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS)

            staleIds = [carId for carId, carObj in carsDict.items() if carObj.lastSeen < frameIndex - 5]
            for carId in staleIds: del carsDict[carId]

            out.write(annotatedFrame)
            cv2.imshow(WINDOW_NAME, annotatedFrame)
            if cv2.waitKey(WAIT_KEY_MS) & 0xFF == EXIT_KEY: break

            if frameIndex % RADAR_STEP_INTERVAL == 0:
                radar_setp = frame_time * RADAR_STEP_INTERVAL
                print(f"Radar step: {radar_setp:.2f}s, Frame index: {frameIndex}")
                radar.step(radar_setp)
                radar.clusterPoints()
                radar.visualizeClusteredStep()
            
            frameIndex += 1

    except Exception as e:
        print(f"Błąd: {e}")
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()