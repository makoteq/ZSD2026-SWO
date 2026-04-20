import os
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from keras import models
from typing import Dict, Final
from pathlib import Path

# Importy lokalne
from algorithms.lane_detection_brute.lane_detection_brute import runLaneDetection
from utils.points import build_lines_equations
from utils.car import Car
from utils.radar import Radar
from utils.utils import drawCustomBox, plotRadarComparison, matchClustersToCars


CURRENT_SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(CURRENT_SCRIPT_PATH, "..", "data"))
VIDEO_PATH = os.path.join(DATA_DIR, "normalTraffic_DistMarkers/rgb(11).mp4")
# VIDEO_PATH = os.path.join(DATA_DIR, "normal_traffic/rgb.mp4")
CSV_PATH = os.path.join(DATA_DIR, "normalTraffic_DistMarkers/overtake.csv")
YOLO_MODEL_PATH = os.path.join(DATA_DIR, "models", "best.pt")
CNN_MODEL_PATH = os.path.join(DATA_DIR, "models", "cnn.h5")
OUTPUT_VIDEO_PATH = os.path.join(DATA_DIR, "output", "trajectory.mp4")

# yolo
FOV = 20.0

START_TIME = 0
CONF_THRESHOLD = 0.8
IMGSZ = 800
ALLOWED_CLASSES_IDS = [0]
MAX_MISSING_FRAMES = 30
LINE_THICKNESS = 1
TRACK_COLOR = (0, 255, 0)

TEXT_COLOR: Final[tuple] = (255, 255, 255)
TEXT_THICKNESS: Final[int] = 2
TEXT_SCALE: Final[float] = 0.7
TEXT_POSITION_X: Final[int] = 20
TEXT_POSITION_Y_START: Final[int] = 30
TEXT_LINE_SPACING: Final[int] = 30

LANE_DEPARTURE_COLOR: Final[tuple] = (0, 0, 255)
ALARM_COLOR: Final[tuple] = (0, 0, 255)
ALARM_SQUARE_SIZE: Final[int] = 40
ALARM_MARGIN: Final[int] = 20
ALARM_ACTIVE = False

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

def activateAlarm():
    global ALARM_ACTIVE
    ALARM_ACTIVE = True
    print("ALARM: Lane departure detected!")


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

            staleIds = [carId for carId, carObj in carsDict.items() if carObj.lastSeen < frameIndex - MAX_MISSING_FRAMES]
            for carId in staleIds: del carsDict[carId]
            
            if frameIndex == 0:
                # # Check if lines.json exists in cache
                project_root = Path(__file__).resolve().parents[1]
                cached_lanes_path = project_root / "data" / "output" / "lines.json"
                print(f"cached_lanes_path = {cached_lanes_path}")
                if cached_lanes_path.exists():
                    lines_path = cached_lanes_path
                else:
                    lines_path = runLaneDetection(videoName="lines_det.mp4",showVideo=True,PASSES_COUNT=6)

                detected_lines = build_lines_equations(lines_path)
                # lines = getManualLaneLines(VIDEO_PATH)
                # detected_lines = [{'m': -0.608171, 'b': 763.608171, 'x_bot': 106.783658, 'abs_m': 0.608171}, {'m': 0.605019, 'b': 1157.789963, 'x_bot': 1811.210037, 'abs_m': 0.605019}]
                # detected_lines = [{'m': -0.7025392986698912, 'b': 1002.1221281741233, 'x_bot': 243.37968561064088, 'abs_m': 0.7025392986698912}, {'m': 0.5273390036452005, 'b': 925.2199270959903, 'x_bot': 1494.746051032807, 'abs_m': 0.5273390036452005}]
                # detected_lines = [{'m': -0.735399284862932, 'b': 1010.0250297973778, 'x_bot': 215.79380214541118, 'abs_m': 0.735399284862932}, {'m': 0.5978520286396182, 'b': 914.3090692124105, 'x_bot': 1559.989260143198, 'abs_m': 0.5978520286396182}]
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
                dist = matchClustersToCars(carsDict, clusterCenters, frameIndex)
                
           
            results = model.track(source=frame, imgsz=IMGSZ, conf=CONF_THRESHOLD,persist=True, verbose=False, device=0 if device == 'cuda' else 'cpu',tracker='bytetrack.yaml', classes=ALLOWED_CLASSES_IDS) 
            annotatedFrame = frame.copy()
            # TODO handle it via arg or smth
            # Draw lines for testing
            
            LANE_LINE_COLOR = (0, 200, 255)
            LANE_LINE_THICKNESS = 2
            y_top = 0
            y_bottom = frameHeight - 1
            for line in detected_lines:
                m = line.get('m')
                b = line.get('b')
                if m is None or b is None:
                    continue

                x_top = int((m * y_top) + b)
                x_bottom = int((m * y_bottom) + b)
                cv2.line(
                    annotatedFrame,
                    (x_top, y_top),
                    (x_bottom, y_bottom),
                    LANE_LINE_COLOR,
                    LANE_LINE_THICKNESS,
                )


            if results[0].boxes.id is not None:
                boxesXyxy = results[0].boxes.xyxy.cpu().numpy()
                boxesXywh = results[0].boxes.xywh.cpu().numpy()
                trackIds = results[0].boxes.id.int().cpu().tolist()
                confidences = results[0].boxes.conf.cpu().tolist()

                for boxXyxy, boxXywh, trackId, conf in zip(boxesXyxy, boxesXywh, trackIds, confidences):
                    if trackId not in carsDict:
                        carsDict[trackId] = Car(trackId)

                    car = carsDict[trackId]

                    car.posGlobalYDifference = dist
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

                    laneDepartureDetected = car.updateLaneState(detected_lines, frameIndex)
                    if laneDepartureDetected:
                        activateAlarm()
                        print(f"[ALERT] Lane departure | ID={trackId} | frame={frameIndex}")

                    if car.isOutsideLane:
                        print(f"[OUTSIDE_LANE] ID={trackId} | frame={frameIndex}")

                    drawCustomBox(annotatedFrame, boxXyxy, trackId, conf, car.type, car.pos[-1].x, car.pos[-1].y, car.velo[-1].v)

                    if car.isOutsideLane and car.laneDepartureFrame >= 0:
                        x1, y1, _, _ = map(int, boxXyxy)
                        cv2.putText(
                            annotatedFrame,
                            "LANE DEPARTURE",
                            (x1, max(20, y1 - 12)),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            LANE_DEPARTURE_COLOR,
                            2,
                        )

                    points = np.array(car.history).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotatedFrame, [points], False, TRACK_COLOR, LINE_THICKNESS)


                ##TODO dodać algorytm wykrywania niebezpieczeństwa

            currentTime = START_TIME + (frameIndex * frame_time)
            cv2.putText(annotatedFrame, f"Frame: {frameIndex}", (TEXT_POSITION_X, TEXT_POSITION_Y_START), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS)
            cv2.putText(annotatedFrame, f"Time: {currentTime:.2f}s", (TEXT_POSITION_X, TEXT_POSITION_Y_START + TEXT_LINE_SPACING), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS)

            if ALARM_ACTIVE:
                x2 = frameWidth - ALARM_MARGIN
                y1 = ALARM_MARGIN
                x1 = x2 - ALARM_SQUARE_SIZE
                y2 = y1 + ALARM_SQUARE_SIZE
                cv2.rectangle(annotatedFrame, (x1, y1), (x2, y2), ALARM_COLOR, -1)

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

