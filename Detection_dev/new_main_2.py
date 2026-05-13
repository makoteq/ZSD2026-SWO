import os
import json
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from typing import Dict, Final
from pathlib import Path
from tqdm import tqdm
from typing import Final, List
import itertools

# Importy lokalne
from algorithms.lane_detection.lane_detector import LaneDetector
from utils.points import build_lines_equations
from utils.car import Car, stoppingDistance
from utils.radar import SENSOR_PITCH_DEG, SENSOR_YAW_DEG, Radar
from utils.depth_v2 import DepthV2
from utils.utils import drawCustomBox, plotRadarComparison, matchClustersToCars, getManualLaneLines, save_car_to_csv, \
    plotYOffsetCorrelation
import matplotlib.pyplot as plt

CURRENT_SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
SCENARIO = "output"
DATA_DIR = os.path.abspath(os.path.join(CURRENT_SCRIPT_PATH, "..", "data"))

# Control
VIDEO_PATH = os.path.join(DATA_DIR, "dataset/wyprzedzanie/wyprzedzanie1.mp4")
RADAR_CSV_PATH = os.path.join(DATA_DIR, "dataset/wyprzedzanie/wyprzedzanie1.csv")

# Speeding
# VIDEO_PATH = os.path.join(DATA_DIR, "alarm/speeding1/rgb.mp4")
# CSV_PATH = os.path.join(DATA_DIR, "alarm/speeding1/radar_points_world.csv")

# overtaking
# VIDEO_PATH = os.path.join(DATA_DIR, "alarm/overtaking1/rgb.mp4")
# CSV_PATH = os.path.join(DATA_DIR, "alarm/overtaking1/radar_points_world.csv")

# overtaking
# VIDEO_PATH = os.path.join(DATA_DIR, "alarm/overtaking2/video_day(4).mp4")
# CSV_PATH = os.path.join(DATA_DIR, "normalTraffic_DistMarkers/radar_points_world.csv")

# Lane departure
# VIDEO_PATH = os.path.join(DATA_DIR, "alarm/trajectory_change1/rgb.mp4")
# CSV_PATH = os.path.join(DATA_DIR, "normalTraffic_DistMarkers/radar_points_world.csv")


CSV_PATH = os.path.join(DATA_DIR, SCENARIO, "car.csv")
YOLO_MODEL_PATH = os.path.join(DATA_DIR, "models", "best.pt")
OUTPUT_VIDEO_PATH = os.path.join(DATA_DIR, "output", "trajectory.mp4")
DEPTH_MODEL_PATH = os.path.join(DATA_DIR, "models", "depth_anything_v2_vits.pth")
DEPTH_LIB_PATH = os.path.join(DATA_DIR, "models", "Depth-Anything-V2")
DEPTH_OUTPUT_DIR = os.path.join(DATA_DIR, "output", "depth_maps")
LINES_JSON_PATH = os.path.join(DATA_DIR, SCENARIO, "lines.json")

# yolo
ROAD_WIDTH_METERS = 7.0
FOV = 14.0

START_TIME = 0.0
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
MASK_Z_MAX = 100.0
MASK_Y_MIN = 0.0
MASK_Y_MAX = 120.0

WINDOW_NAME = "Traffic Analysis"
WAIT_KEY_MS = 1
EXIT_KEY = ord('q')

WATCHDOG_ZONE_MIN = 20.0
WATCHDOG_ZONE_MAX = 60.0
OVERTAKING_MARGIN = 0.0

tracked_pairs = {}
overtaking_events = []

if __name__ == "__main__":

    # correctionFunc = plotYOffsetCorrelation(CSV_PATH)

    correctionFunc = lambda x: 0.0
    model = YOLO(YOLO_MODEL_PATH)
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
    detector = LaneDetector()

    try:
        while cap.isOpened():

            success, frame = cap.read()
            if not success: break

            staleIds = [carId for carId, carObj in carsDict.items() if carObj.lastSeen < frameIndex - 5]
            for carId in staleIds: del carsDict[carId]

            if frameIndex == 0:
                if os.path.exists(LINES_JSON_PATH):
                    with open(LINES_JSON_PATH, 'r') as f:
                        lines_dick = json.load(f)
                else:
                    lines_dick = detector.detect(frame)
                    with open(LINES_JSON_PATH, 'w') as f:
                        json.dump(lines_dick, f, indent=4)

                xLeft = lines_dick["left_line"]["start"][0]
                xRight = lines_dick["right_line"]["start"][0]
                road_width_h0_px = abs(xRight - xLeft)

                detected_lines = [lines_dick["left_line"], lines_dick["right_line"]]

                print("Detected lines:", detected_lines)

            if frameIndex % RADAR_STEP_INTERVAL == 0:
                radar_setp = frame_time * RADAR_STEP_INTERVAL
                radar.step(radar_setp)
                radar.clusterPoints()
                # radar.visualizeClusteredStep()
                clusterCenters = radar.getClusterCenters()
                dist, carMap = matchClustersToCars(carsDict, clusterCenters, frameIndex)

                for clusterId, cluster in enumerate(clusterCenters):
                    currentDistance: float = abs(cluster['y_corrected'])
                    carId = carMap.get(clusterId)
                    print(f"Distance: {currentDistance:.2f} m")
                    currentVelocity: float = abs(cluster['radial_velocity'])
                    if currentVelocity > SPEED_LIMIT:
                        print(f"[WARNING] Speed limit exceeded by cluster: {currentVelocity:.2f} m/s")
                    elif carId is not None and carId in carsDict:
                        car = carsDict[carId]
                        stoppingDist = car.stoppingDistance[-1].distance

                        if stoppingDist >= currentDistance:
                            print(
                                f"[LEVEL 2 WARNING] Detected vehicle won't be able to stop in time: {stoppingDist:.2f} m > {currentDistance:.2f} m")

                # TODO przkeorczenuie prędkosci

                plotRadarComparison(radar.minX, radar.maxX, 0, radar.maxY, carsDict, clusterCenters)
                print(dist)

            results = model.track(source=frame, imgsz=IMGSZ, conf=CONF_THRESHOLD, persist=True, verbose=False,
                                  device=0 if device == 'cuda' else 'cpu', tracker='bytetrack.yaml',
                                  classes=ALLOWED_CLASSES_IDS)
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

                    car.update(
                        boxXywh,
                        conf,
                        frame,
                        frameIndex,
                        detected_lines,
                        road_width_h0_px,
                        FOV,
                        frame_time,
                        IMGSZ,
                        radar,
                        correctionFunc
                    )

                    drawCustomBox(annotatedFrame, boxXyxy, trackId, conf, car.pos[-1].x, car.pos[-1].y,
                                  car.cameraDistance,
                                  car.velo[-1].v, car.stoppingDistance[-1].distance, )

                    save_car_to_csv(car, trackId, frameIndex, CSV_PATH)

                    points = np.array(car.history).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotatedFrame, [points], False, TRACK_COLOR, LINE_THICKNESS)

                # overtake WATCHDOG
                cars_in_zone = []
                for car_id, car in carsDict.items():
                    if car.cameraDistance is not None and WATCHDOG_ZONE_MIN <= car.cameraDistance <= WATCHDOG_ZONE_MAX:
                        cars_in_zone.append(car)

                current_zone_ids = {car.trackId for car in cars_in_zone}

                for car1, car2 in itertools.combinations(cars_in_zone, 2):
                    # Sort IDs to ensure consistent ordering of pairs
                    id_pair = tuple(sorted((car1.trackId, car2.trackId)))
                    dist1 = car1.cameraDistance
                    dist2 = car2.cameraDistance

                    # Establish leader and follower based on distance
                    current_leader = car1.trackId if dist1 > dist2 else car2.trackId
                    current_follower = car2.trackId if dist1 > dist2 else car1.trackId
                    distance_diff = abs(dist1 - dist2)

                    if id_pair not in tracked_pairs:
                        # Register the pair for the first time if they are close enough
                        if distance_diff > OVERTAKING_MARGIN:
                            tracked_pairs[id_pair] = current_leader
                    else:
                        previous_leader = tracked_pairs[id_pair]

                        # Detect overtaking if the leader has changed and they are sufficiently apart
                        if previous_leader != current_leader and distance_diff > OVERTAKING_MARGIN:
                            msg = f"WYPRZEDZANIE: {current_leader} wyprzedzil {current_follower}"
                            print(f"[WATCHDOG] {msg}")
                            # Add event to display for a few frames
                            overtaking_events.append({'msg': msg, 'frames': int(fps * 2)})
                            
                            # Update the tracked leader for this pair
                            tracked_pairs[id_pair] = current_leader

                # Remove pairs that are no longer in the zone
                tracked_pairs = {pair: leader for pair, leader in tracked_pairs.items() 
                                 if pair[0] in current_zone_ids and pair[1] in current_zone_ids}

                # Draw overtaking events
                y_offset = TEXT_POSITION_Y_START + 3 * TEXT_LINE_SPACING
                for event in overtaking_events[:]:
                    cv2.putText(annotatedFrame, event['msg'], (TEXT_POSITION_X, y_offset),
                                cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, (0, 0, 255), TEXT_THICKNESS)
                    y_offset += TEXT_LINE_SPACING
                    event['frames'] -= 1
                    if event['frames'] <= 0:
                        overtaking_events.remove(event)
                

            currentTime = START_TIME + (frameIndex * frame_time)
            cv2.putText(annotatedFrame, f"Frame: {frameIndex}", (TEXT_POSITION_X, TEXT_POSITION_Y_START),
                        cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS)
            cv2.putText(annotatedFrame, f"Time: {currentTime:.2f}s",
                        (TEXT_POSITION_X, TEXT_POSITION_Y_START + TEXT_LINE_SPACING), cv2.FONT_HERSHEY_SIMPLEX,
                        TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS)

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


