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
from utils.depth_v2 import DepthV2
from utils.utils import  drawCustomBox, plotRadarComparison, matchClustersToCars, getManualLaneLines, save_car_to_csv, plotYOffsetCorrelation
import matplotlib.pyplot as plt


CURRENT_SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
SCENERIO = "normalTraffic_DistMarkers"
DATA_DIR = os.path.abspath(os.path.join(CURRENT_SCRIPT_PATH, "..", "data"))
VIDEO_PATH = os.path.join(DATA_DIR,SCENERIO, "rgb.mp4")
RADAR_CSV_PATH = os.path.join(DATA_DIR,SCENERIO, "radar_points_world.csv")
CSV_PATH = os.path.join(DATA_DIR,SCENERIO, "car.csv")
YOLO_MODEL_PATH = os.path.join(DATA_DIR, "models", "best.pt")
CNN_MODEL_PATH = os.path.join(DATA_DIR, "models", "cnn.h5")
OUTPUT_VIDEO_PATH = os.path.join(DATA_DIR, "output", "trajectory.mp4")
DEPTH_MODEL_PATH = os.path.join(DATA_DIR, "models", "Depth-Anything-V2", "depth_anything_v2_vits.pth")
DEPTH_LIB_PATH = os.path.join(DATA_DIR, "models", "Depth-Anything-V2")
DEPTH_OUTPUT_DIR = os.path.join(DATA_DIR, "output", "depth_maps")


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

SPEED_LIMIT_KMH: Final[float] = 60.0
SPEED_LIMIT: Final[float] = SPEED_LIMIT_KMH / 3.6
# radar
RADAR_STEP_INTERVAL = 10
MASK_Z_MIN = 30.0
MASK_Z_MAX = 50.0
MASK_Y_MIN = 75
MASK_Y_MAX = 130.0


WINDOW_NAME = "Traffic Analysis"
WAIT_KEY_MS = 1
EXIT_KEY = ord('q')

if __name__ == "__main__":
    
    # correctionFunc = plotYOffsetCorrelation(CSV_PATH)

    correctionFunc = lambda x: 0.0
    model = YOLO(YOLO_MODEL_PATH)
    cnn = models.load_model(CNN_MODEL_PATH, compile=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        exit(1)

    radar: Radar = Radar(RADAR_CSV_PATH, START_TIME)
    radar.applyMask(MASK_Z_MIN, MASK_Z_MAX, MASK_Y_MIN, MASK_Y_MAX)
    radar.addNoise()
    radar.findLane()
    radar.visualize()



    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"FPS: {fps}")
    frame_time = 1.0 / fps
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(START_TIME * fps))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, int(fps), (frameWidth, frameHeight))

    depthProcessor = DepthV2(modelPath=DEPTH_MODEL_PATH, libPath=DEPTH_LIB_PATH)

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame = cap.read()[1]
    baseDepthMap = depthProcessor.getDepthMap(frame)
    depthProcessor.saveDepthMap(baseDepthMap, DEPTH_OUTPUT_DIR, name="base_depth")

    carsDict: Dict[int, Car] = {}
    frameIndex = 0

    try:
        while cap.isOpened():
            
            success, frame = cap.read()
            if not success: break

            staleIds = [carId for carId, carObj in carsDict.items() if carObj.lastSeen < frameIndex - 5]
            for carId in staleIds: del carsDict[carId]
            
            if frameIndex == 0:
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

                for cluster in clusterCenters:
                    currentVelocity: float = abs(cluster['radial_velocity'])
                    if currentVelocity > SPEED_LIMIT:
                        print(f"[WARNING] Speed limit exceeded by cluster: {currentVelocity:.2f} m/s")

                #TODO przkeorczenuie prędkosci 


                plotRadarComparison(radar.minX, radar.maxX, 0, radar.maxY, carsDict, clusterCenters)
                dist  = matchClustersToCars(carsDict, clusterCenters, frameIndex)
                print(dist)
                
            results = model.track(source=frame, imgsz=IMGSZ, conf=CONF_THRESHOLD,persist=True, verbose=False, device=0 if device == 'cuda' else 'cpu',tracker='bytetrack.yaml', classes=ALLOWED_CLASSES_IDS) 
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
                        FOV,
                        frame_time,
                        IMGSZ,
                        radar,
                        correctionFunc
                    )

                    drawCustomBox(annotatedFrame, boxXyxy, trackId, conf, car.type, car.pos[-1].x, car.pos[-1].y, car.velo[-1].v)

                    save_car_to_csv(car, trackId, frameIndex, CSV_PATH)

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


