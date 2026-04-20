import os
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from keras import models
from typing import Dict, Final
from pathlib import Path
from tqdm import tqdm
from typing import Final, List

# Importy lokalne
from algorithms.lane_detection_brute.lane_detection_brute import runLaneDetection
from utils.points import build_lines_equations
from utils.car import Car
from utils.radar import SENSOR_PITCH_DEG, SENSOR_YAW_DEG, Radar
from utils.utils import  drawCustomBox, plotRadarComparison, matchClustersToCars, getManualLaneLines
import matplotlib.pyplot as plt


CURRENT_SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(CURRENT_SCRIPT_PATH, "..", "data"))
VIDEO_PATH = os.path.join(DATA_DIR, "normalTraffic_DistMarkers/rgb.mp4")
# VIDEO_PATH = os.path.join(DATA_DIR, "normal_traffic/rgb.mp4")
CSV_PATH = os.path.join(DATA_DIR, "normalTraffic_DistMarkers/radar_points_world.csv")
YOLO_MODEL_PATH = os.path.join(DATA_DIR, "models", "best.pt")
CNN_MODEL_PATH = os.path.join(DATA_DIR, "models", "cnn.h5")
OUTPUT_VIDEO_PATH = os.path.join(DATA_DIR, "output", "trajectory.mp4")

# yolo
ROAD_WIDTH_METERS = 7.0
FOV = 20.0

START_TIME = 12
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

BOX_COLOR: Final[tuple] = (0, 255, 0)
BOX_THICKNESS: Final[int] = 2

# radar
RADAR_STEP_INTERVAL = 10
MASK_Z_MIN = 30.0
MASK_Z_MAX = 50.0
MASK_Y_MIN = 75
MASK_Y_MAX = 130.0

# window
WINDOW_NAME = "Traffic Analysis"
WAIT_KEY_MS = 1
EXIT_KEY = ord('q')

if __name__ == "__main__":
    model = YOLO(YOLO_MODEL_PATH)
    cnn = models.load_model(CNN_MODEL_PATH, compile=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        exit(1)

    radar: Radar = Radar(CSV_PATH, START_TIME)
    radar.applyMask(MASK_Z_MIN, MASK_Z_MAX, MASK_Y_MIN, MASK_Y_MAX)
    radar.findLane()

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

    try:
        while cap.isOpened():
            
            success, frame = cap.read()
            if not success: break

            staleIds = [carId for carId, carObj in carsDict.items() if carObj.lastSeen < frameIndex - 5]
            for carId in staleIds: del carsDict[carId]
            

            if frameIndex == 0:
                # # Check if lines.json exists in cache
                # project_root = Path(__file__).resolve().parents[1]
                # cached_lanes_path = project_root / "data" / "output" / "lines.json"
                # print(f"cached_lanes_path = {cached_lanes_path}")
                # if cached_lanes_path.exists():
                #     lines_path = cached_lanes_path
                # elif cached_lanes_path.exists():
                #     lines_path = cached_lanes_path
                # else:
                #     lines_path = runLaneDetection(showVideo=False,PASSES_COUNT=12)

                # detected_lines = build_lines_equations(lines_path)
                lines = getManualLaneLines(VIDEO_PATH)
                detected_lines = [{'m': -0.608171, 'b': 763.608171, 'x_bot': 106.783658, 'abs_m': 0.608171}, {'m': 0.605019, 'b': 1157.789963, 'x_bot': 1811.210037, 'abs_m': 0.605019}]
                y=0
                xLeft = (detected_lines[0]['m'] * y) + detected_lines[0]['b']
                xRight = (detected_lines[1]['m'] * y) + detected_lines[1]['b']
                road_width_h0_px = abs(xRight - xLeft)

            if frameIndex % RADAR_STEP_INTERVAL == 0:
                radar_setp = frame_time * RADAR_STEP_INTERVAL
                radar.step(radar_setp)
                radar.clusterPoints()
                # radar.visualizeClusteredStep()
                clusterCenters = radar.getClusterCenters()
                
                #cluster centers to już są samochody 
                #TODO przkeorczenuie prędkosci 

                plotRadarComparison(radar.minX, radar.maxX, 0, radar.maxY, carsDict, clusterCenters)
                matchClustersToCars(carsDict, clusterCenters, frameIndex)
                
           
            results = model.track(source=frame, imgsz=IMGSZ, conf=CONF_THRESHOLD,persist=True, verbose=False, device=0 if device == 'cuda' else 'cpu',tracker='bytetrack.yaml', classes=ALLOWED_CLASSES_IDS) 
            annotatedFrame = frame.copy()

            # TODO handle it via arg or smth
            # Draw lines for testing
            
            # LANE_LINE_COLOR = (0, 200, 255)
            # LANE_LINE_THICKNESS = 2
            # y_top = 0
            # y_bottom = frameHeight - 1
            # for line in detected_lines:
            #     m = line.get('m')
            #     b = line.get('b')
            #     if m is None or b is None:
            #         continue

            #     x_top = int((m * y_top) + b)
            #     x_bottom = int((m * y_bottom) + b)
            #     cv2.line(
            #         annotatedFrame,
            #         (x_top, y_top),
            #         (x_bottom, y_bottom),
            #         LANE_LINE_COLOR,
            #         LANE_LINE_THICKNESS,
            #     )


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
                        FOV,
                        frame_time,
                        IMGSZ,
                        radar
                    )

                    drawCustomBox(annotatedFrame, boxXyxy, trackId, conf, car.type, car.pos[-1].x, car.pos[-1].y, car.velo[-1].v)

                    points = np.array(car.history).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotatedFrame, [points], False, TRACK_COLOR, LINE_THICKNESS)


                ##TODO dodać algorytm wykrywania niebezpieczeństwa

            currentTime = START_TIME + (frameIndex * frame_time)
            cv2.putText(annotatedFrame, f"Frame: {frameIndex}", (TEXT_POSITION_X, TEXT_POSITION_Y_START), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS)
            cv2.putText(annotatedFrame, f"Time: {currentTime:.2f}s", (TEXT_POSITION_X, TEXT_POSITION_Y_START + TEXT_LINE_SPACING), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS)

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

