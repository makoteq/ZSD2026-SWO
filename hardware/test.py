import os
import sys
import re
import json
import cv2
import subprocess
from ultralytics import YOLO
from typing import Dict
from typing import Final, List
import queue
from threading import Thread

from utils.car import Car
from utils.radar import Radar
from utils.depth_v2 import loadOrComputeDepthMap, rankCarsByDepth, flattenRowsMedianBackground, saveDepthVisualization
from utils.weather import calcStoppingDistance
from utils.utils import detectOvertaking, drawCustomBox, save_car_to_csv
from utils.alarm_manager import AlarmManager
import psutil
import csv
import time
from datetime import date
import numpy as np


CURRENT_SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.abspath(os.path.join(CURRENT_SCRIPT_PATH, "..", "data"))

def resolve_config_path(path: str, base_dir: str = DATA_DIR) -> str:
    if not isinstance(path, str):
        raise ValueError(f"Config path must be a string: {path!r}")

    normalized = path.replace('\\', '/').strip()
    windows_drive = re.match(r'^[A-Za-z]:/', normalized)
    if windows_drive:
        normalized = normalized.split(':', 1)[1]
        normalized = normalized.lstrip('/')

    if os.path.isabs(normalized):
        return os.path.abspath(normalized)

    return os.path.abspath(os.path.join(base_dir, normalized))

class RTSPVideoGetter:
    def __init__(self, capture_object):
        self.stream = capture_object
        self.stopped = False
        self.Q = queue.Queue(maxsize=4)

    def start(self):
        t = Thread(target=self.update, args=())
        t.daemon = True
        t.start()
        return self

    def update(self):
        while True:
            if self.stopped:
                return
            
            if not self.Q.full():
                (grabbed, frame) = self.stream.read()
                if not grabbed:
                    self.stop()
                    return
                self.Q.put(frame)
            else:
                time.sleep(0.005)

    def read(self):
        if self.stopped and self.Q.empty():
            return None
        try:
            return self.Q.get(timeout=2.0)
        except queue.Empty:
            return None

    def is_opened(self):
        return self.stream.isOpened() and not self.stopped

    def stop(self):
        self.stopped = True
        if self.stream.isOpened():
            self.stream.release()

SCENARIO = "output"

CONFIG_JSON_PATH = os.path.join(DATA_DIR, "config", "config.json")
if not os.path.exists(CONFIG_JSON_PATH):
    print("config.json not found")
    subprocess.run([sys.executable, os.path.join(CURRENT_SCRIPT_PATH, "setup.py")], check=True)

with open(CONFIG_JSON_PATH, 'r', encoding='utf-8') as f:
    cfg = json.load(f)


VIDEO_PATH = os.path.join(DATA_DIR, "dataset/alarm/2_AB/2_AB.mp4")
RADAR_CSV_PATH = os.path.join(DATA_DIR, "dataset/alarm/2_AB/2_AB--6.0.csv")
CSV_PATH = os.path.join(DATA_DIR, SCENARIO, "car.csv")
YOLO_MODEL_PATH = os.path.join(DATA_DIR, "models", "416_latest_full_integer_quant_edgetpu.tflite")
OUTPUT_VIDEO_PATH = os.path.join(DATA_DIR, "output", f"{os.path.splitext(os.path.basename(VIDEO_PATH))[0]}_trajectory.mp4")

DEPTH_MODEL_PATH = resolve_config_path(cfg["depth"]["model_path"])
DEPTH_LIB_PATH = os.path.join(DATA_DIR, "models", "Depth-Anything-V2")
DEPTH_OUTPUT_DIR = resolve_config_path(cfg["depth"]["output_dir"])
NPY_PATH = resolve_config_path(cfg["depth"]["npy_path"])

ROAD_WIDTH_METERS = cfg["geometry"]["road_width_meters"]
FOV = cfg["geometry"]["fov_deg"]

START_TIME = 0.0
RADAR_DELAY = 0.0
CONF_THRESHOLD = 0.8
IMGSZ = 416
ALLOWED_CLASSES_IDS = [0]
LINE_THICKNESS = 1
TRACK_COLOR = (0, 255, 0)
LANE_DEPARTURE_MARGIN_PX = 10

TEXT_COLOR: Final[tuple] = (255, 255, 255)
TEXT_THICKNESS: Final[int] = 2
TEXT_SCALE: Final[float] = 0.7
TEXT_POSITION_X: Final[int] = 20
TEXT_POSITION_Y_START: Final[int] = 30
TEXT_LINE_SPACING: Final[int] = 30

SPEED_LIMIT_KMH: Final[float] = 60.0
SPEED_LIMIT: Final[float] = SPEED_LIMIT_KMH / 3.6

RADAR_STEP_INTERVAL = 15
MASK_Z_MIN = 30.0
MASK_Z_MAX = 100.0
MASK_Y_MIN = 0.0
MASK_Y_MAX = 120.0

ACTUAL_RADAR_OFFSET= 8.0
RADAR_DEBUG = False

WINDOW_NAME = "Traffic Analysis"
DISPLAY_SCALE = 0.5 
WAIT_KEY_MS = 1
EXIT_KEY = ord('q')

METRICS_DIR = os.path.join(DATA_DIR, "metrics")
os.makedirs(METRICS_DIR, exist_ok=True)
output_csv = os.path.join(METRICS_DIR, "metryki_new_main_metrics.csv")

alarm_manager = AlarmManager()

if __name__ == "__main__":

    SAVE_VIDEO: Final[bool] = False        
    SAVE_CSV: Final[bool] = False 
    SHOW_PREVIEW: Final[bool] = False

    correctionFunc = lambda x: 0.0
    model = YOLO(YOLO_MODEL_PATH)
    
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

    frame_time = 1.0 / fps
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(START_TIME * fps))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    os.makedirs(os.path.dirname(OUTPUT_VIDEO_PATH), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, int(fps), (frameWidth, frameHeight))

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(START_TIME * fps))
    _, firstFrame = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(START_TIME * fps))
    
    depthMapComputed = not os.path.exists(NPY_PATH)
    baseDepthMap = loadOrComputeDepthMap(NPY_PATH, firstFrame, DEPTH_MODEL_PATH, DEPTH_LIB_PATH, DEPTH_OUTPUT_DIR)

    firstFrameResults = model.predict(source=firstFrame, imgsz=IMGSZ,iou= 0.4, conf=CONF_THRESHOLD, verbose=False, classes=ALLOWED_CLASSES_IDS)
    firstFrameBboxes = [
        {'x1': box[0], 'y1': box[1], 'x2': box[2], 'y2': box[3]}
        for box in firstFrameResults[0].boxes.xyxy.cpu().numpy()
    ] if firstFrameResults[0].boxes is not None else []
    baseDepthMap = flattenRowsMedianBackground(baseDepthMap, firstFrameBboxes, paddingFactor=0.05)
    # if depthMapComputed:
    #     saveDepthVisualization(baseDepthMap, DEPTH_OUTPUT_DIR, name="base_depth_filled")

    video_getter = RTSPVideoGetter(cap).start()

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

    unique_vehicle_ids = set()
    metrics_data = []
    warmup_frames = 30  
    print(f"Warmup frames configuration: {warmup_frames}")
    total_processing_time = 0.0
    processed_frames_count = 0
    real_fps = 0.0

    last_cpu_percent = 0.0
    last_ram_percent = 0.0

    if SHOW_PREVIEW:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1280, 720)

    try:
        while video_getter.is_opened():
            
            frame = video_getter.read()
            if frame is None: break

            t_start = time.time() 

            currentTime = START_TIME + (frameIndex * frame_time)

            staleIds = [carId for carId, carObj in carsDict.items() if carObj.lastSeen < frameIndex - 5]
            for carId in staleIds: del carsDict[carId]

            if frameIndex % RADAR_STEP_INTERVAL == 0:
                radar_setup = frame_time * RADAR_STEP_INTERVAL
                radar.step(radar_setup)
                radar.clusterPoints()
                clusterCenters = radar.getClusterCenters()

                if clusterCenters:
                    closestRadarCluster = min(clusterCenters, key=lambda c: abs(c['y_corrected']))
                    closestRadarDistance = abs(closestRadarCluster['y_corrected']) - ACTUAL_RADAR_OFFSET

                    if closestRadarDistance <= OVERTAKING_LINE_THRESHOLD and overtakingLineY is None:
                        overtakingLineTriggered = True

                for cluster in clusterCenters:
                    currentDistance: float = abs(cluster['y_corrected']) - ACTUAL_RADAR_OFFSET
                    currentVelocity: float = abs(cluster['radial_velocity'])
                    stoppingDistance: float = abs(calcStoppingDistance(currentVelocity, multiplier))
                    if RADAR_DEBUG:
                        print(f"Distance: {currentDistance:.2f}, adjusted with offset {ACTUAL_RADAR_OFFSET:.1f} m")

                    if stoppingDistance >= currentDistance:
                        alarm_manager.trigger(2, f"Detected vehicle won't be able to stop in time {stoppingDistance:.1f}m > {currentDistance:.1f}m",
                                              0.0, currentTime)
                    elif currentVelocity > SPEED_LIMIT:
                        alarm_manager.trigger(1, f"Speed limit exceeded by cluster: {currentVelocity:.1f} m/s", 0.0,
                                              currentTime)

         
            results = model.track(source=frame, imgsz=IMGSZ, iou=0.4, conf=CONF_THRESHOLD, persist=True, verbose=False, tracker='bytetrack.yaml', classes=ALLOWED_CLASSES_IDS)
            current_boxes = results[0].boxes
            inference_ms = results[0].speed.get('inference', 0.0)

            if current_boxes is not None and current_boxes.id is not None:
                boxesXyxy = current_boxes.xyxy.cpu().numpy()
                boxesXywh = current_boxes.xywh.cpu().numpy()
                trackIds = current_boxes.id.int().cpu().tolist()
                confidences = current_boxes.conf.cpu().tolist()

                for i, (boxXyxy, boxXywh, trackId, conf) in enumerate(zip(boxesXyxy, boxesXywh, trackIds, confidences)):
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
      

                    if trackId != "N/A":
                        unique_vehicle_ids.add(trackId)

                    #if not is_detection_frame: 
                        # drawCustomBox(
                        #     annotatedFrame, 
                        #     boxXyxy, 
                        #     trackId, 
                        #     conf, 
                        #     car.pos[-1].x, 
                        #     car.pos[-1].y,
                        #     car.velo[-1].v, 
                        #     car.stoppingDistance[-1].distance
                        # )

                    if SAVE_CSV:
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

                            out_of_left = x1 < (x_left_line_at_y2 - LANE_DEPARTURE_MARGIN_PX)
                            out_of_right = x2 > (x_right_line_at_y2 + LANE_DEPARTURE_MARGIN_PX)

                            if out_of_left or out_of_right:
                                direction = "LEFT" if out_of_left else "RIGHT"

                                if frameIndex - laneDepartureCooldown.get(trackId,
                                                                          -LANE_DEPARTURE_COOLDOWN_FRAMES) >= LANE_DEPARTURE_COOLDOWN_FRAMES:
                                    alarm_manager.trigger(1, f"Anomaly by the {direction} line!",
                                                          0.0, currentTime)
                                    laneDepartureCooldown[trackId] = frameIndex

                if overtakingLineTriggered and overtakingLineY is None:
                    closestCarIdx = max(range(len(trackIds)), key=lambda i: boxesXyxy[i][3])
                    overtakingLineY = int(boxesXyxy[closestCarIdx][3])
                    print(f"[MONITOR] Overtaking zone starts from Y={overtakingLineY}px ({closestRadarDistance:.2f}m)")

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
                            print(msg) 
                            alarm_manager.trigger(1, msg, 0.0, currentTime)
                            overtakeCooldown[pair] = frameIndex

                prevZoneRanking = currentZoneRanking

            # cv2.putText(annotatedFrame, f"Frame: {frameIndex}", (TEXT_POSITION_X, TEXT_POSITION_Y_START), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS)
            # cv2.putText(annotatedFrame, f"Time: {currentTime:.2f}s", (TEXT_POSITION_X, TEXT_POSITION_Y_START + TEXT_LINE_SPACING), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS)
            # cv2.putText(annotatedFrame, f"FPS: {real_fps:.2f}", (TEXT_POSITION_X, TEXT_POSITION_Y_START + 2 * TEXT_LINE_SPACING), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS)
            
            # alarm_manager.draw(annotatedFrame, currentTime)
            
            if SAVE_VIDEO:
                out.write(frame)

            t_end = time.time()
            frame_duration = t_end - t_start 
            real_fps = 1.0 / frame_duration if frame_duration > 0 else 0.0

            frame_time_ms = frame_duration * 1000.0
            num_detections = len(current_boxes) if current_boxes is not None else 0
            avg_confidence = float(np.mean([box.conf for box in current_boxes])) if num_detections > 0 else 0.0
            if frameIndex > warmup_frames:
                total_processing_time += frame_duration
                processed_frames_count += 1

                if frameIndex % 30 == 0:
                    last_cpu_percent = psutil.cpu_percent()
                    last_ram_percent = psutil.virtual_memory().percent

                metrics_data.append({
                    'Model': os.path.basename(YOLO_MODEL_PATH),
                    'Slice_Size': IMGSZ, 
                    'Frame_Time_ms': round(frame_time_ms, 2),
                    'Instant_FPS': round(real_fps, 2),
                    'Inference_ms': round(inference_ms, 2),
                    'Objects_On_Frame': num_detections,
                    'Unique_Objects_Total': len(unique_vehicle_ids),
                    'Avg_Confidence': round(avg_confidence, 3),
                    'CPU_Percent': last_cpu_percent,
                    'RAM_Percent': last_ram_percent
                })

            if SHOW_PREVIEW:
                cv2.imshow(WINDOW_NAME, cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
                if cv2.waitKey(1) & 0xFF == ord('q'): break

            frameIndex += 1

    except Exception as e:
        print(f"Error: {e}")
    finally:
        video_getter.stop() 
        out.release()
        cv2.destroyAllWindows()
 
        if metrics_data:
            with open(output_csv, mode='w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=metrics_data[0].keys())
                writer.writeheader()
                writer.writerows(metrics_data)
            print(f"\nmetrics saved to: {output_csv}")
            
            if total_processing_time > 0 and processed_frames_count > 0:
                global_fps = processed_frames_count / total_processing_time
                avg_frame_time = (total_processing_time / processed_frames_count) * 1000.0
                
                print("\n" + "="*40)
                print("       GLOBAL PERFORMANCE METRICS (after warmup)          ")
                print("="*40)
                print(f" Number of processed frames (after warmup): {processed_frames_count}")
                print(f" Total processing time:                 {total_processing_time:.2f} s")
                print(f" Average frame time:                    {avg_frame_time:.2f} ms")
                print(f" Actual Average FPS:           {global_fps:.2f} FPS")
                print("="*40 + "\n")
            print(f"Total number of detected unique vehicles: {len(unique_vehicle_ids)}")
