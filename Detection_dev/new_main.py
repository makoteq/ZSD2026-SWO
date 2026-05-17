import os
import sys
import json
import torch
import cv2
import subprocess
from ultralytics import YOLO
from typing import Dict
from typing import Final, List

# Local imports
from utils.car import Car
from utils.radar import Radar
from utils.depth_v2 import loadOrComputeDepthMap, rankCarsByDepth, flattenRowsMedianBackground, saveDepthVisualization
from utils.weather import calcStoppingDistance
from utils.utils import detectOvertaking, drawCustomBox, save_car_to_csv
from utils.alarm_manager import AlarmManager

CURRENT_SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
SCENARIO = "output"
DATA_DIR = os.path.abspath(os.path.join(CURRENT_SCRIPT_PATH, "..", "data"))

CONFIG_JSON_PATH = os.path.join(DATA_DIR, "config", "config.json")
if not os.path.exists(CONFIG_JSON_PATH):
    print("config.json not found")
    subprocess.run([sys.executable, os.path.join(CURRENT_SCRIPT_PATH, "setup.py")], check=True)

with open(CONFIG_JSON_PATH, 'r', encoding='utf-8') as f:
    cfg = json.load(f)

# Control
VIDEO_PATH = os.path.join(DATA_DIR, "dataset/noalarm/1_Control/1_Control.mp4")
RADAR_CSV_PATH = os.path.join(DATA_DIR, "dataset/noalarm/1_Control/1_Control--6.0.csv")

# normal traffic
# VIDEO_PATH = os.path.join(DATA_DIR, "dataset/noalarm/7_Control/7_Control.mp4")
# RADAR_CSV_PATH = os.path.join(DATA_DIR, "dataset/noalarm/7_Control/7_Control--6.0.csv")

# speeding
# VIDEO_PATH = os.path.join(DATA_DIR, "dataset/alarm/1_A/1_A.mp4")
# RADAR_CSV_PATH = os.path.join(DATA_DIR, "dataset/alarm/1_A/1_A--6.0.csv")

# speeding/lane departure
# VIDEO_PATH = os.path.join(DATA_DIR, "dataset/alarm/1_AB/1_AB.mp4")
# RADAR_CSV_PATH = os.path.join(DATA_DIR, "dataset/alarm/1_AB/1_AB--6.0.csv")

# speeding/overtaking
# VIDEO_PATH = os.path.join(DATA_DIR, "dataset/alarm/1_AC/1_AC.mp4")
# RADAR_CSV_PATH = os.path.join(DATA_DIR, "dataset/alarm/1_AC/1_AC--6.0.csv")

# lane departure
# VIDEO_PATH = os.path.join(DATA_DIR, "dataset/alarm/1_B/1_B.mp4")
# RADAR_CSV_PATH = os.path.join(DATA_DIR, "dataset/alarm/1_B/1_B--6.0.csv")

# overtaking
# VIDEO_PATH = os.path.join(DATA_DIR, "dataset/alarm/1_C/1_C.mp4")
# RADAR_CSV_PATH = os.path.join(DATA_DIR, "dataset/alarm/1_C/1_C--6.0.csv")

CSV_PATH = os.path.join(DATA_DIR, SCENARIO, "car.csv")
YOLO_MODEL_PATH = os.path.join(DATA_DIR, "models", "best.pt")
OUTPUT_VIDEO_PATH = os.path.join(DATA_DIR, "output", "trajectory.mp4")
DEPTH_MODEL_PATH = cfg["depth"]["model_path"]
DEPTH_LIB_PATH = os.path.join(DATA_DIR, "models", "Depth-Anything-V2")
DEPTH_OUTPUT_DIR = cfg["depth"]["output_dir"]
NPY_PATH = cfg["depth"]["npy_path"]

# yolo
ROAD_WIDTH_METERS = cfg["geometry"]["road_width_meters"]
FOV = cfg["geometry"]["fov_deg"]

START_TIME = 0.0
RADAR_DELAY = 0.0
CONF_THRESHOLD = 0.8
IMGSZ = 800
ALLOWED_CLASSES_IDS = [0]
LINE_THICKNESS = 1
TRACK_COLOR = (0, 255, 0)

TEXT_COLOR: Final[tuple] = (255, 255, 255)
TEXT_THICKNESS: Final[int] = 2
TEXT_SCALE: Final[float] = 0.7
TEXT_POSITION_X: Final[int] = 20
TEXT_POSITION_Y_START: Final[int] = 30
TEXT_LINE_SPACING: Final[int] = 30

SPEED_LIMIT_KMH: Final[float] = 60.0
SPEED_LIMIT: Final[float] = SPEED_LIMIT_KMH / 3.6

# radar
RADAR_STEP_INTERVAL = 10
MASK_Z_MIN = 30.0
MASK_Z_MAX = 100.0
MASK_Y_MIN = 0.0
MASK_Y_MAX = 120.0

ACTUAL_RADAR_OFFSET= 8.0
RADAR_DEBUG = True

WINDOW_NAME = "Traffic Analysis"
DISPLAY_SCALE = 0.5 # TEMP WINDOW
WAIT_KEY_MS = 1
EXIT_KEY = ord('q')


alarm_manager = AlarmManager()

if __name__ == "__main__":
    
    # correctionFunc = plotYOffsetCorrelation(CSV_PATH)

    correctionFunc = lambda x: 0.0
    model = YOLO(YOLO_MODEL_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    with open(RADAR_CSV_PATH, 'r') as f:
        f.readline()  
        first_line = f.readline().split(',')
        RADAR_DELAY = float(first_line[0])
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        exit(1)
    print(f"Radar delay: {RADAR_DELAY}")
    radar: Radar = Radar(RADAR_CSV_PATH, START_TIME + RADAR_DELAY)
    radar.debug = RADAR_DEBUG
    radar.applyMask(MASK_Z_MIN, MASK_Z_MAX, MASK_Y_MIN, MASK_Y_MAX)
    radar.addNoise()
    radar.findLane()
    if RADAR_DEBUG:
        radar.visualize()

    fps = cap.get(cv2.CAP_PROP_FPS)
    print(f"FPS: {fps}")

    print(f"{cfg["weather"]["latitude"],cfg["weather"]["longitude"]}\nDate: {cfg["weather"]["date"]}, "
          f"Weather conditions: {cfg["weather"]["condition"]} \n{cfg["weather"]["description"]}")

    frame_time = 1.0 / fps
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(START_TIME * fps))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, int(fps), (frameWidth, frameHeight))

    # Depth map for the first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(START_TIME * fps))
    _, firstFrame = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(START_TIME * fps))
    
    depthMapComputed = not os.path.exists(NPY_PATH)
    baseDepthMap = loadOrComputeDepthMap(NPY_PATH, firstFrame, DEPTH_MODEL_PATH, DEPTH_LIB_PATH, DEPTH_OUTPUT_DIR)

    firstFrameResults = model.predict(source=firstFrame, imgsz=IMGSZ, conf=CONF_THRESHOLD, verbose=False, device=0 if device == 'cuda' else 'cpu', classes=ALLOWED_CLASSES_IDS)
    firstFrameBboxes = [
        {'x1': box[0], 'y1': box[1], 'x2': box[2], 'y2': box[3]}
        for box in firstFrameResults[0].boxes.xyxy.cpu().numpy()
    ] if firstFrameResults[0].boxes is not None else []
    baseDepthMap = flattenRowsMedianBackground(baseDepthMap, firstFrameBboxes, paddingFactor=0.05)
    if depthMapComputed:
        saveDepthVisualization(baseDepthMap, DEPTH_OUTPUT_DIR, name="base_depth_filled")

    carsDict: Dict[int, Car] = {}
    frameIndex = 0

    overtakingLineY = None
    overtakingLineTriggered = False
    OVERTAKING_LINE_THRESHOLD = 60.0
    closestRadarDistance = float('inf')

    prevZoneRanking: List[int] = []
    overtakeCooldown: Dict[frozenset, int] = {}
    OVERTAKE_COOLDOWN_FRAMES = int(3.0 * fps)

    laneDepartureCooldown: Dict[int, int] = {}
    LANE_DEPARTURE_COOLDOWN_FRAMES = int(3.0 * fps)

    lines_dick = cfg["lanes"]
    xLeft = lines_dick["left_line"]["start"][0]
    xRight = lines_dick["right_line"]["start"][0]
    road_width_h0_px = abs(xRight - xLeft)
    detected_lines = [lines_dick["left_line"], lines_dick["right_line"]]

    multiplier = cfg["weather"]["Markiplier"]

    print("Detected lines:", detected_lines)

    try:
        while cap.isOpened():
            
            success, frame = cap.read()
            if not success: break

            currentTime = START_TIME + (frameIndex * frame_time)

            staleIds = [carId for carId, carObj in carsDict.items() if carObj.lastSeen < frameIndex - 5]
            for carId in staleIds: del carsDict[carId]

            if frameIndex % RADAR_STEP_INTERVAL == 0:
                radar_setup = frame_time * RADAR_STEP_INTERVAL
                radar.step(radar_setup)
                radar.clusterPoints()
                # radar.visualizeClusteredStep()
                clusterCenters = radar.getClusterCenters()

                # trigger when first ccluster enter overtaking zone
                if clusterCenters:
                    closestRadarCluster = min(clusterCenters, key=lambda c: abs(c['y_corrected']))
                    closestRadarDistance = abs(closestRadarCluster['y_corrected']) - ACTUAL_RADAR_OFFSET

                    if closestRadarDistance <= OVERTAKING_LINE_THRESHOLD and overtakingLineY is None:
                        overtakingLineTriggered = True

                for cluster in clusterCenters:
                    currentDistance: float = abs(cluster['y_corrected']) -ACTUAL_RADAR_OFFSET
                    currentVelocity: float = abs(cluster['radial_velocity'])
                    stoppingDistance: float = abs(calcStoppingDistance(currentVelocity, multiplier))
                    print(f"Distance: {currentDistance:.2f}, adjusted with offset {ACTUAL_RADAR_OFFSET:.1f} m")
                    if RADAR_DEBUG:
                        print(f"Distance: {currentDistance:.2f}, adjusted with offset {ACTUAL_RADAR_OFFSET:.1f} m")

                    if stoppingDistance >= currentDistance:
                        alarm_manager.trigger(2,f"Detected vehicle won't be able to stop in time {stoppingDistance:.1f}m > {currentDistance:.1f}m",
                                              0.0, currentTime)
                    elif currentVelocity > SPEED_LIMIT:
                        alarm_manager.trigger(1, f"Speed limit exceeded by cluster: {currentVelocity:.1f} m/s", 0.0,
                                              currentTime)
                #TODO przkeorczenuie prędkosci

                
            results = model.track(source=frame, imgsz=IMGSZ, conf=CONF_THRESHOLD,persist=True, verbose=False, device=0 if device == 'cuda' else 'cpu',tracker='bytetrack.yaml', classes=ALLOWED_CLASSES_IDS) 
            annotatedFrame = frame.copy()

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
                        detected_lines,
                        road_width_h0_px,
                        FOV,
                        frame_time,
                        IMGSZ,
                        radar,
                        correctionFunc
                    )

                    drawCustomBox(
                        annotatedFrame,
                        boxXyxy,
                        trackId,
                        conf,
                        car.pos[-1].x,
                        car.pos[-1].y,
                        car.velo[-1].v,
                        car.stoppingDistance[-1].distance,
                    )

                    save_car_to_csv(car, trackId, frameIndex, CSV_PATH)

                    # --- LANE DEPARTURE WATCHDOG ---
                    x1, y1, x2, y2 = boxXyxy

                    if len(detected_lines) >= 2:
                        left_line = detected_lines[0]
                        right_line = detected_lines[1]

                        m_left, b_left = left_line.get('m'), left_line.get('b')
                        m_right, b_right = right_line.get('m'), right_line.get('b')

                        if None not in (m_left, b_left, m_right, b_right):
                            x_left_line_at_y2 = (m_left * y2) + b_left
                            x_right_line_at_y2 = (m_right * y2) + b_right

                            out_of_left = x1 < x_left_line_at_y2
                            out_of_right = x2 > x_right_line_at_y2

                            if out_of_left or out_of_right:
                                direction = "LEFT" if out_of_left else "RIGHT"

                                if frameIndex - laneDepartureCooldown.get(trackId,
                                                                          -LANE_DEPARTURE_COOLDOWN_FRAMES) >= LANE_DEPARTURE_COOLDOWN_FRAMES:
                                    alarm_manager.trigger(1, f"Lane departure car {trackId} crossed {direction} line!",
                                                          0.0, currentTime)
                                    laneDepartureCooldown[trackId] = frameIndex
                    # --- END LANE DEPARTURE WATCHDOG ---

                    # points = np.array(car.history).astype(np.int32).reshape((-1, 1, 2))
                    # cv2.polylines(annotatedFrame, [points], False, TRACK_COLOR, LINE_THICKNESS)


                # OVERTAKING
                 
                # determining overtaking zone line position
                if overtakingLineTriggered and overtakingLineY is None:
                    closestCarIdx = max(range(len(trackIds)), key=lambda i: boxesXyxy[i][3])
                    overtakingLineY = int(boxesXyxy[closestCarIdx][3])
                    print(f"[MONITOR] Overtaking zone starts from Y={overtakingLineY}px ({closestRadarDistance:.2f}m)")

                # list of cars that are in the overtaking zone, below overtakingLineY
                cars_in_zone = [
                    {'id': tid, 'x1': int(box[0]), 'y1': int(box[1]), 'x2': int(box[2]), 'y2': int(box[3])}
                    for tid, box in zip(trackIds, boxesXyxy)
                    if overtakingLineY is not None and int(box[1]) >= overtakingLineY
                ]
                
                ranked = rankCarsByDepth(baseDepthMap, cars_in_zone)
                currentZoneRanking = [car['id'] for car in ranked]
                
                if prevZoneRanking:
                    overtakes = detectOvertaking(prevZoneRanking, currentZoneRanking)
                    for overtaker, overtaken in overtakes:
                        pair = frozenset([overtaker, overtaken])
                        if frameIndex - overtakeCooldown.get(pair, -OVERTAKE_COOLDOWN_FRAMES) >= OVERTAKE_COOLDOWN_FRAMES:
                            msg = f"[OVERTAKE] Frame {frameIndex}: Car {overtaker} overtook Car {overtaken}"
                            print(msg) #debug print
                            alarm_manager.trigger(1, msg, 0.0, currentTime)
                            overtakeCooldown[pair] = frameIndex

                prevZoneRanking = currentZoneRanking

            # overtaking line visualization 
            # if overtakingLineY is not None:
            #     cv2.line(annotatedFrame, (0, overtakingLineY), (frameWidth, overtakingLineY), (0, 255, 255), 2)

            cv2.putText(annotatedFrame, f"Frame: {frameIndex}", (TEXT_POSITION_X, TEXT_POSITION_Y_START), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS)
            cv2.putText(annotatedFrame, f"Time: {currentTime:.2f}s", (TEXT_POSITION_X, TEXT_POSITION_Y_START + TEXT_LINE_SPACING), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS)

            alarm_manager.draw(annotatedFrame, currentTime)
            out.write(annotatedFrame)

            #cv2.imshow(WINDOW_NAME, annotatedFrame)
            # TEMP WINDOW
            previewFrame = cv2.resize(annotatedFrame, None, fx = DISPLAY_SCALE, fy = DISPLAY_SCALE, interpolation = cv2.INTER_AREA)
            cv2.imshow(WINDOW_NAME, previewFrame)
            if cv2.waitKey(WAIT_KEY_MS) & 0xFF == EXIT_KEY: break


            frameIndex += 1

    except Exception as e:
        print(f"Error: {e}")
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()


