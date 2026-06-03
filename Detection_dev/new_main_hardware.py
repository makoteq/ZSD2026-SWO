import os
import sys
import json
import cv2
import subprocess
import psutil
import csv
import time
import numpy as np
import re
from typing import Dict, Final, List

from ultralytics import YOLO
from utils.car import Car
from utils.radar import Radar
from utils.depth_v2 import loadOrComputeDepthMap, rankCarsByDepth, flattenRowsMedianBackground, saveDepthVisualization
from utils.weather import calcStoppingDistance
from utils.utils import detectOvertaking, drawCustomBox, save_car_to_csv
from utils.alarm_manager import AlarmManager

CURRENT_SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(CURRENT_SCRIPT_PATH, "..", "data"))

def resolve_config_path(path: str, base_dir: str = DATA_DIR) -> str:
    if not isinstance(path, str):
        raise ValueError(f"Config path must be a string: {path!r}")
    normalized = path.replace('\\', '/').strip()
    windows_drive = re.match(r'^[A-Za-z]:/', normalized)
    if windows_drive:
        normalized = normalized.split(':', 1)[1].lstrip('/')
    if os.path.isabs(normalized): return os.path.abspath(normalized)
    return os.path.abspath(os.path.join(base_dir, normalized))

CONFIG_JSON_PATH = os.path.join(DATA_DIR, "config", "config.json")
if not os.path.exists(CONFIG_JSON_PATH):
    print("config.json not found, running setup...")
    subprocess.run([sys.executable, os.path.join(CURRENT_SCRIPT_PATH, "setup.py")], check=True)

with open(CONFIG_JSON_PATH, 'r', encoding='utf-8') as f:
    cfg = json.load(f)

SCENARIO = "output"
VIDEO_PATH = os.path.join(DATA_DIR, "dataset/alarm/2_AB/2_AB.mp4")
RADAR_CSV_PATH = os.path.join(DATA_DIR, "dataset/alarm/2_AB/2_AB--6.0.csv")
CSV_PATH = os.path.join(DATA_DIR, SCENARIO, "car.csv")
YOLO_MODEL_PATH = os.path.join(DATA_DIR, "models", "416_latest_full_integer_quant_edgetpu.tflite")
OUTPUT_VIDEO_PATH = os.path.join(DATA_DIR, "output", f"{os.path.splitext(os.path.basename(VIDEO_PATH))[0]}_trajectory.mp4")

DEPTH_MODEL_PATH = resolve_config_path(cfg["depth"]["model_path"])
DEPTH_LIB_PATH = os.path.join(DATA_DIR, "models", "Depth-Anything-V2")
DEPTH_OUTPUT_DIR = resolve_config_path(cfg["depth"]["output_dir"])
NPY_PATH = resolve_config_path(cfg["depth"]["npy_path"])

METRICS_DIR = os.path.join(DATA_DIR, "metrics")
os.makedirs(METRICS_DIR, exist_ok=True)
output_csv = os.path.join(METRICS_DIR, "metryki_new_main_metrics.csv")

SHOW_PREVIEW = True 

ROAD_WIDTH_METERS = cfg["geometry"]["road_width_meters"]
FOV = cfg["geometry"]["fov_deg"]
START_TIME = 0.0
CONF_THRESHOLD = 0.8
IMGSZ = 416
ALLOWED_CLASSES_IDS = [0]
LANE_DEPARTURE_MARGIN_PX = 10
SPEED_LIMIT_KMH: Final[float] = 60.0
SPEED_LIMIT: Final[float] = SPEED_LIMIT_KMH / 3.6
OVERTAKING_LINE_THRESHOLD = 60.0

TEXT_COLOR: Final[tuple] = (255, 255, 255)
TEXT_THICKNESS: Final[int] = 2
TEXT_SCALE: Final[float] = 0.7
TEXT_POSITION_X: Final[int] = 20
TEXT_POSITION_Y_START: Final[int] = 30
TEXT_LINE_SPACING: Final[int] = 30

RADAR_STEP_INTERVAL = 40
ACTUAL_RADAR_OFFSET = 8.0
RADAR_DEBUG = True

alarm_manager = AlarmManager()

if __name__ == "__main__":
    correctionFunc = lambda x: 0.0
    model = YOLO(YOLO_MODEL_PATH)
    
    with open(RADAR_CSV_PATH, 'r') as f:
        f.readline()  
        RADAR_DELAY = float(f.readline().split(',')[0])
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened(): exit(1)
    
    radar = Radar(RADAR_CSV_PATH, START_TIME + RADAR_DELAY)
    radar.debug = RADAR_DEBUG
    radar.applyMask(30.0, 100.0, 0.0, 120.0)
    radar.addNoise()
    radar.findLane()

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_time = 1.0 / fps
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    os.makedirs(os.path.dirname(OUTPUT_VIDEO_PATH), exist_ok=True)
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, cv2.VideoWriter_fourcc(*'mp4v'), int(fps), (frameWidth, frameHeight))

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(START_TIME * fps))
    _, firstFrame = cap.read()
    
    depthMapComputed = not os.path.exists(NPY_PATH)
    baseDepthMap = loadOrComputeDepthMap(NPY_PATH, firstFrame, DEPTH_MODEL_PATH, DEPTH_LIB_PATH, DEPTH_OUTPUT_DIR)

    firstFrameResults = model.predict(source=firstFrame, imgsz=IMGSZ, conf=CONF_THRESHOLD, verbose=False, classes=ALLOWED_CLASSES_IDS)
    firstFrameBboxes = [{'x1': box[0], 'y1': box[1], 'x2': box[2], 'y2': box[3]} for box in firstFrameResults[0].boxes.xyxy.cpu().numpy()] if firstFrameResults[0].boxes is not None else []
    baseDepthMap = flattenRowsMedianBackground(baseDepthMap, firstFrameBboxes, paddingFactor=0.05)
    if depthMapComputed: saveDepthVisualization(baseDepthMap, DEPTH_OUTPUT_DIR, name="base_depth_filled")

    carsDict: Dict[int, Car] = {}
    frameIndex = 0
    overtakingLineY = None
    overtakingLineTriggered = False
    closestRadarDistance = float('inf')

    prevZoneRanking: List[int] = []
    overtakeCooldown: Dict[frozenset, int] = {}
    laneDepartureCooldown: Dict[int, int] = {}
    COOLDOWN_FRAMES = int(3.0 * fps)

    lines_dick = cfg["lanes"]
    xLeft = lines_dick["left_line"]["start"][0]
    xRight = lines_dick["right_line"]["start"][0]
    road_width_h0_px = abs(xRight - xLeft)
    detected_lines = [lines_dick["left_line"], lines_dick["right_line"]]
    multiplier = cfg["weather"]["Markiplier"]

    unique_vehicle_ids = set()
    metrics_data = []
    warmup_frames = 30  
    total_processing_time = 0.0
    processed_frames_count = 0
    real_fps = 0.0
    last_results_boxes = None

    if SHOW_PREVIEW:
        WINDOW_NAME = "Traffic Analysis"
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success: break

            t_start = time.time() 
            currentTime = START_TIME + (frameIndex * frame_time)

            staleIds = [carId for carId, carObj in carsDict.items() if carObj.lastSeen < frameIndex - 5]
            for carId in staleIds: del carsDict[carId]

            if frameIndex % RADAR_STEP_INTERVAL == 0:
                radar.step(frame_time * RADAR_STEP_INTERVAL)
                radar.clusterPoints()
                clusterCenters = radar.getClusterCenters()

                if clusterCenters:
                    closestRadarCluster = min(clusterCenters, key=lambda c: abs(c['y_corrected']))
                    closestRadarDistance = abs(closestRadarCluster['y_corrected']) - ACTUAL_RADAR_OFFSET
                    if closestRadarDistance <= OVERTAKING_LINE_THRESHOLD and overtakingLineY is None:
                        overtakingLineTriggered = True

                for cluster in clusterCenters:
                    currentDistance = abs(cluster['y_corrected']) - ACTUAL_RADAR_OFFSET
                    currentVelocity = abs(cluster['radial_velocity'])
                    stoppingDistance = abs(calcStoppingDistance(currentVelocity, multiplier))

                    if stoppingDistance >= currentDistance:
                        alarm_manager.trigger(2, f"Zbyt blisko! Hamowanie: {stoppingDistance:.1f}m > Dystans: {currentDistance:.1f}m", 0.0, currentTime)
                    elif currentVelocity > SPEED_LIMIT:
                        alarm_manager.trigger(1, f"Przekroczenie prędkości: {currentVelocity:.1f} m/s", 0.0, currentTime)

            results = model.track(source=frame, imgsz=IMGSZ, conf=CONF_THRESHOLD, persist=True, verbose=False, tracker='bytetrack.yaml', classes=ALLOWED_CLASSES_IDS)
            last_results_boxes = results[0].boxes
            inference_ms = results[0].speed.get('inference', 0.0)

            if last_results_boxes is not None and last_results_boxes.id is not None:
                boxesXyxy = last_results_boxes.xyxy.cpu().numpy()
                boxesXywh = last_results_boxes.xywh.cpu().numpy()
                trackIds = last_results_boxes.id.int().cpu().tolist()
                confidences = last_results_boxes.conf.cpu().tolist()

                for i, (boxXyxy, boxXywh, trackId, conf) in enumerate(zip(boxesXyxy, boxesXywh, trackIds, confidences)):
                    if trackId not in carsDict:
                        carsDict[trackId] = Car(trackId)

                    car = carsDict[trackId]
                    
                    car.update(boxXywh, conf, frame, frameIndex, detected_lines, road_width_h0_px, FOV, frame_time, IMGSZ, radar, correctionFunc)
                    save_car_to_csv(car, trackId, frameIndex, CSV_PATH)

                    if trackId != "N/A": unique_vehicle_ids.add(trackId)

                    drawCustomBox(frame, boxXyxy, trackId, conf, car.pos[-1].x, car.pos[-1].y, car.velo[-1].v, car.stoppingDistance[-1].distance)

                    if len(detected_lines) >= 2:
                        m_left, b_left = detected_lines[0].get('m'), detected_lines[0].get('b')
                        m_right, b_right = detected_lines[1].get('m'), detected_lines[1].get('b')

                        if None not in (m_left, b_left, m_right, b_right):
                            x1, y1, x2, y2 = boxXyxy
                            x_left_line = (m_left * y2) + b_left
                            x_right_line = (m_right * y2) + b_right

                            if x1 < (x_left_line - LANE_DEPARTURE_MARGIN_PX) or x2 > (x_right_line + LANE_DEPARTURE_MARGIN_PX):
                                direction = "LEFT" if x1 < x_left_line else "RIGHT"
                                if frameIndex - laneDepartureCooldown.get(trackId, -COOLDOWN_FRAMES) >= COOLDOWN_FRAMES:
                                    alarm_manager.trigger(1, f"Linia {direction} przekroczona!", 0.0, currentTime)
                                    laneDepartureCooldown[trackId] = frameIndex

                if overtakingLineTriggered and overtakingLineY is None:
                    closestCarIdx = max(range(len(trackIds)), key=lambda i: boxesXyxy[i][3])
                    overtakingLineY = int(boxesXyxy[closestCarIdx][3])

                cars_in_zone = [{'id': tid, 'x1': int(box[0]), 'y1': int(box[1]), 'x2': int(box[2]), 'y2': int(box[3])} 
                                for tid, box in zip(trackIds, boxesXyxy) if overtakingLineY is not None and int(box[1]) >= overtakingLineY]
                
                if cars_in_zone:
                    ranked = rankCarsByDepth(baseDepthMap, cars_in_zone)
                    currentZoneRanking = [c['id'] for c in ranked]
                    if prevZoneRanking:
                        for overtaker, overtaken in detectOvertaking(prevZoneRanking, currentZoneRanking):
                            pair = frozenset([overtaker, overtaken])
                            if frameIndex - overtakeCooldown.get(pair, -COOLDOWN_FRAMES) >= COOLDOWN_FRAMES:
                                alarm_manager.trigger(1, f"Wyprzedzanie: {overtaker} wyprzedził {overtaken}", 0.0, currentTime)
                                overtakeCooldown[pair] = frameIndex
                    prevZoneRanking = currentZoneRanking

            cv2.putText(frame, f"Frame: {frameIndex}", (TEXT_POSITION_X, TEXT_POSITION_Y_START), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS)
            cv2.putText(frame, f"Time: {currentTime:.2f}s", (TEXT_POSITION_X, TEXT_POSITION_Y_START + TEXT_LINE_SPACING), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS)
            cv2.putText(frame, f"FPS: {real_fps:.2f}", (TEXT_POSITION_X, TEXT_POSITION_Y_START + 2 * TEXT_LINE_SPACING), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, TEXT_COLOR, TEXT_THICKNESS)
            
            alarm_manager.draw(frame, currentTime)
            out.write(frame)

            t_end = time.time()
            frame_duration = t_end - t_start 
            real_fps = 1.0 / frame_duration if frame_duration > 0 else 0.0

            if frameIndex > warmup_frames:
                total_processing_time += frame_duration
                processed_frames_count += 1
                metrics_data.append({
                    'Model': os.path.basename(YOLO_MODEL_PATH),
                    'Slice_Size': IMGSZ, 
                    'Frame_Time_ms': round(frame_duration * 1000.0, 2),
                    'Instant_FPS': round(real_fps, 2),
                    'Inference_ms': round(inference_ms, 2),
                    'Objects_On_Frame': len(last_results_boxes) if last_results_boxes is not None else 0,
                    'Unique_Objects_Total': len(unique_vehicle_ids),
                    'Avg_Confidence': round(float(np.mean(last_results_boxes.conf.cpu().numpy())), 3) if (last_results_boxes is not None and len(last_results_boxes) > 0) else 0.0,
                    'CPU_Percent': psutil.cpu_percent(),
                    'RAM_Percent': psutil.virtual_memory().percent
                })

            if SHOW_PREVIEW:
                cv2.imshow(WINDOW_NAME, cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA))
                if cv2.waitKey(1) & 0xFF == ord('q'): break

            frameIndex += 1

    except Exception as e:
        print(f"Błąd: {e}")
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()
 
        if metrics_data:
            with open(output_csv, mode='w', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=metrics_data[0].keys())
                writer.writeheader()
                writer.writerows(metrics_data)
            
            if total_processing_time > 0 and processed_frames_count > 0:
                print("\n" + "="*40)
                print("       GLOBAL PERFORMANCE METRICS          ")
                print("="*40)
                print(f" Processed frames:      {processed_frames_count}")
                print(f" Total processing time: {total_processing_time:.2f} s")
                print(f" Average frame time:    {(total_processing_time / processed_frames_count) * 1000.0:.2f} ms")
                print(f" Actual Average FPS:    {(processed_frames_count / total_processing_time):.2f} FPS")
                print("="*40 + "\n")