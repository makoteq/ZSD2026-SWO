import os
import json
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from typing import Dict, Final, Tuple, Optional
from pathlib import Path
from tqdm import tqdm
from typing import List
import matplotlib.pyplot as plt


# Local imports
from algorithms.lane_detection.lane_detector import LaneDetector
#from utils.points import build_lines_equations
from utils.car import Car
from utils.radar import SENSOR_PITCH_DEG, SENSOR_YAW_DEG, Radar
from utils.depth_v2 import DepthV2, loadOrComputeDepthMap, rankCarsByDepth, flattenRowsMedianBackground, saveDepthVisualization
from datetime import date
from utils.weather import calcStoppingDistance, getWeather
from utils.utils import detectOvertaking, drawCustomBox, plotRadarComparison, save_car_to_csv, \
    plotYOffsetCorrelation
from utils.alarm_manager import AlarmManager
from batch_runner import (
    build_batch_report_path,
    discover_batch_recordings,
    format_alarm_reasons,
    save_batch_report_markdown,
)

CURRENT_SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_SCRIPT_PATH, ".."))
SCENARIO = "output"
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
REPORTS_DIR = os.path.join(PROJECT_ROOT, "docs", "Reports")

# Control
# VIDEO_PATH = os.path.join(DATA_DIR, "dataset/noalarm/1_control.mp4")
# RADAR_CSV_PATH = os.path.join(DATA_DIR, "dataset/noalarm/1_control.csv")

# seb new overtaking
VIDEO_PATH = os.path.join(DATA_DIR, "new_overtaking/1_overtaking/1_overtaking.mp4")
RADAR_CSV_PATH = os.path.join(DATA_DIR, "new_overtaking/1_overtaking/1_overtaking--6.5.csv")

# # seb new double overtaking
# VIDEO_PATH = os.path.join(DATA_DIR, "dataset/new_overtaking/1_overtaking_double/1_overtaking_double.mp4")
# RADAR_CSV_PATH = os.path.join(DATA_DIR, "dataset/new_overtaking/1_overtaking_double/1_overtaking_double--6.5.csv")

# overtaking
# VIDEO_PATH = os.path.join(DATA_DIR, "dataset/alarm/overtaking1/rgb.mp4")
# RADAR_CSV_PATH = os.path.join(DATA_DIR, "dataset/alarm/overtaking1/radar_points_world.csv")

# Speeding
# VIDEO_PATH = os.path.join(DATA_DIR, "dataset/alarm/21_speeding.mp4")
# RADAR_CSV_PATH = os.path.join(DATA_DIR, "dataset/alarm/21_speeding.csv")

# overtaking
# VIDEO_PATH = os.path.join(DATA_DIR, "dataset/alarm/33_overtaking.mp4")
# RADAR_CSV_PATH = os.path.join(DATA_DIR, "dataset/alarm/33_overtaking.csv")

# overtaking
# VIDEO_PATH = os.path.join(DATA_DIR, "dataset/alarm/overtaking2/video_day(4).mp4")
# RADAR_CSV_PATH = os.path.join(DATA_DIR, "dataset/normalTraffic_DistMarkers/radar_points_world.csv")

# # Lane departure
# VIDEO_PATH = os.path.join(DATA_DIR, "dataset/11_pull_over/11_pull_over.mp4")
# RADAR_CSV_PATH = os.path.join(DATA_DIR, "dataset/11_pull_over/11_pull_over--14.csv")

# Test
# VIDEO_PATH = os.path.join(DATA_DIR, "dataset/test/test6.mp4")
# RADAR_CSV_PATH = os.path.join(DATA_DIR, "dataset/test/test6.csv")

CSV_PATH = os.path.join(DATA_DIR, SCENARIO, "car.csv")
YOLO_MODEL_PATH = os.path.join(DATA_DIR, "models", "best.pt")
OUTPUT_VIDEO_PATH = os.path.join(DATA_DIR, "output", "trajectory.mp4")
DEPTH_MODEL_PATH = os.path.join(DATA_DIR, "models", "depth_anything_v2_vits.pth")
DEPTH_LIB_PATH = os.path.join(DATA_DIR, "models", "Depth-Anything-V2")
DEPTH_OUTPUT_DIR = os.path.join(DATA_DIR, "output", "depth_maps")
NPY_PATH = os.path.join(DEPTH_OUTPUT_DIR, "base_depth.npy")
LINES_JSON_PATH = os.path.join(DATA_DIR, SCENARIO, "lines.json") # always delete lines.json file if video path is changed


# yolo
ROAD_WIDTH_METERS = 7.0
FOV = 20.0

START_TIME = 0.0
RADAR_DELAY = 0.0
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

ACTUAL_RADAR_OFFSET= 8.0

WINDOW_NAME = "Traffic Analysis"
DISPLAY_SCALE = 0.5 # TEMP WINOW
WAIT_KEY_MS = 1
EXIT_KEY = ord('q')

RUN_MODE = "batch"  # "single" or "batch"
DATASET_ROOT = os.path.join(DATA_DIR, "dataset")
BATCH_CSV_SUFFIX = "--6.0.csv"
BATCH_OUTPUT_DIR = os.path.join(DATA_DIR, "output", "batch")
BATCH_WRITE_VIDEOS = False
BATCH_SHOW_WINDOW = True
BATCH_DEBUG_VISUALS = False
BATCH_MAX_RECORDINGS: Optional[int] = None


def classify_alarm_reason(reason: str) -> str:
    if not reason:
        return "unknown"
    lowered = reason.lower()
    if "speed limit" in lowered:
        return "speed_limit_exceeded"
    if "overtake" in lowered:
        return "overtaking_detected"
    if "lane departure" in lowered:
        return "lane_departure"
    if "stop" in lowered and "distance" in lowered:
        return "stopping_distance"
    return "other"


CODE_ORDER: Final[Tuple[str, ...]] = ("A", "B", "C")
CODE_TO_REASON: Final[Dict[str, str]] = {
    "A": "speed_limit_exceeded",
    "B": "lane_departure",
    "C": "overtaking_detected",
}
REASON_TO_CODE: Final[Dict[str, str]] = {
    "speed_limit_exceeded": "A",
    "lane_departure": "B",
    "overtaking_detected": "C",
}


def parse_expected_codes_from_path(video_path: str) -> List[str]:
    stem = Path(video_path).stem
    parts = stem.split("_", 1)
    if len(parts) < 2:
        return []
    raw_codes = parts[1].upper()
    codes = [code for code in raw_codes if code in CODE_TO_REASON]
    return sorted(set(codes), key=CODE_ORDER.index)


def parse_detected_codes_from_reasons(alarm_reasons: str) -> List[str]:
    if not alarm_reasons or alarm_reasons == "none":
        return []
    codes: List[str] = []
    for reason_part in alarm_reasons.split(";"):
        reason = reason_part.split("(", 1)[0].strip()
        code = REASON_TO_CODE.get(reason)
        if code:
            codes.append(code)
    return sorted(set(codes), key=CODE_ORDER.index)


def format_codes(codes: List[str]) -> str:
    return "".join(codes) if codes else "none"


class BatchAlarmTracker:
    def __init__(self) -> None:
        self.manager = AlarmManager()
        self.reason_counts: Dict[str, int] = {}
        self.active = False

    def trigger(self, level: int, reason: str, disable_radar_duration: float, current_time: float) -> None:
        self.active = True
        key = classify_alarm_reason(reason)
        self.reason_counts[key] = self.reason_counts.get(key, 0) + 1
        self.manager.trigger(level, reason, disable_radar_duration, current_time)

    def is_radar_disabled(self, current_time: float) -> bool:
        return self.manager.is_radar_disabled(current_time)

    def draw(self, frame, current_time: float) -> None:
        self.manager.draw(frame, current_time)

    def format_reasons(self) -> str:
        return format_alarm_reasons(self.reason_counts)


def build_output_video_path(video_path: str, base_output_dir: str, prefix: str = "") -> str:
    stem = Path(video_path).stem
    safe_prefix = f"{prefix}_" if prefix else ""
    file_name = f"{safe_prefix}{stem}_annotated.mp4"
    os.makedirs(base_output_dir, exist_ok=True)
    return os.path.join(base_output_dir, file_name)


def run_single_recording(
    model: YOLO,
    device: str,
    video_path: str,
    radar_csv_path: str,
    output_video_path: Optional[str],
    car_csv_path: Optional[str],
    lines_json_path: str,
    depth_output_dir: str,
    npy_path: str,
    show_window: bool,
    write_output: bool,
    debug_visuals: bool,
    precomputed_lines: Optional[dict] = None,
    precomputed_road_width_px: Optional[float] = None,
    precomputed_overtaking_line_y: Optional[int] = None,
    precomputed_depth_map: Optional[np.ndarray] = None,
    weather: Optional[object] = None,
) -> Tuple[bool, str, Optional[dict], Optional[float], Optional[int], Optional[np.ndarray]]:
    correctionFunc = lambda x: 0.0

    with open(radar_csv_path, 'r') as f:
        f.readline()
        first_line = f.readline().split(',')
        radar_delay = float(first_line[0])

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    alarm_tracker = BatchAlarmTracker()

    print(f"Radar delay: {radar_delay}")
    radar: Radar = Radar(radar_csv_path, START_TIME + radar_delay)
    radar.applyMask(MASK_Z_MIN, MASK_Z_MAX, MASK_Y_MIN, MASK_Y_MAX)
    radar.addNoise()
    radar.findLane()
    if debug_visuals:
        radar.visualize()

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise RuntimeError(f"Invalid FPS for video: {video_path}")
    print(f"FPS: {fps}")

    lat = 54.37163
    lon = 18.61898
    Weather = weather or getWeather(lat, lon, date.today(), 12)
    print(f"{lat,lon}\nDate: {date.today()}, Weather conditions: {Weather.condition} \n{Weather.description}")

    frame_time = 1.0 / fps
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(START_TIME * fps))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = None
    if write_output and output_video_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, int(fps), (frameWidth, frameHeight))

    # Depth map for the first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(START_TIME * fps))
    _, firstFrame = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(START_TIME * fps))

    os.makedirs(depth_output_dir, exist_ok=True)
    if precomputed_depth_map is not None:
        baseDepthMap = precomputed_depth_map
        depthMapComputed = False
    else:
        depthMapComputed = not os.path.exists(npy_path)
        baseDepthMap = loadOrComputeDepthMap(npy_path, firstFrame, DEPTH_MODEL_PATH, DEPTH_LIB_PATH, depth_output_dir)

    firstFrameResults = model.predict(
        source=firstFrame,
        imgsz=IMGSZ,
        conf=CONF_THRESHOLD,
        verbose=False,
        device=0 if device == 'cuda' else 'cpu',
        classes=ALLOWED_CLASSES_IDS,
    )
    firstFrameBboxes = [
        {'x1': box[0], 'y1': box[1], 'x2': box[2], 'y2': box[3]}
        for box in firstFrameResults[0].boxes.xyxy.cpu().numpy()
    ] if firstFrameResults[0].boxes is not None else []
    baseDepthMap = flattenRowsMedianBackground(baseDepthMap, firstFrameBboxes, paddingFactor=0.05)
    if depthMapComputed:
        saveDepthVisualization(baseDepthMap, depth_output_dir, name="base_depth_filled")

    carsDict: Dict[int, Car] = {}
    frameIndex = 0

    overtakingLineY = precomputed_overtaking_line_y
    overtakingLineTriggered = False
    OVERTAKING_LINE_THRESHOLD = 60.0
    closestRadarDistance = float('inf')

    prevZoneRanking: List[int] = []
    overtakeCooldown: Dict[frozenset, int] = {}
    OVERTAKE_COOLDOWN_FRAMES = int(3.0 * fps)

    laneDepartureCooldown: Dict[int, int] = {}
    LANE_DEPARTURE_COOLDOWN_FRAMES = int(3.0 * fps)

    detector = LaneDetector()

    try:
        while cap.isOpened():

            success, frame = cap.read()
            if not success:
                break

            currentTime = START_TIME + (frameIndex * frame_time)

            staleIds = [carId for carId, carObj in carsDict.items() if carObj.lastSeen < frameIndex - 5]
            for carId in staleIds:
                del carsDict[carId]

            if frameIndex == 0:
                # lane detection only once (reuse precomputed lines across batch)
                if precomputed_lines is not None:
                    lines_dick = precomputed_lines
                elif os.path.exists(lines_json_path):
                    with open(lines_json_path, 'r') as f:
                        lines_dick = json.load(f)
                else:
                    lines_dick = detector.detect(frame)
                    os.makedirs(os.path.dirname(lines_json_path), exist_ok=True)
                    with open(lines_json_path, 'w') as f:
                        json.dump(lines_dick, f, indent=4)

                if precomputed_road_width_px is not None:
                    road_width_h0_px = precomputed_road_width_px
                else:
                    xLeft = lines_dick["left_line"]["start"][0]
                    xRight = lines_dick["right_line"]["start"][0]
                    road_width_h0_px = abs(xRight - xLeft)

                # we are using only left and right lines
                detected_lines = [lines_dick["left_line"], lines_dick["right_line"]] # converting dictionary into array for easier use

                print("Detected lines:", detected_lines)

            if frameIndex % RADAR_STEP_INTERVAL == 0:
                radar_setp = frame_time * RADAR_STEP_INTERVAL
                radar.step(radar_setp)
                radar.clusterPoints()
                # radar.visualizeClusteredStep()
                clusterCenters = radar.getClusterCenters()

                # trigger when first ccluster enter overtaking zone
                if clusterCenters:
                    closestRadarCluster = min(clusterCenters, key=lambda c: abs(c['y_corrected']))
                    closestRadarDistance = abs(closestRadarCluster['y_corrected']) -ACTUAL_RADAR_OFFSET

                    if closestRadarDistance <= OVERTAKING_LINE_THRESHOLD and overtakingLineY is None:
                        overtakingLineTriggered = True

                for cluster in clusterCenters:
                    currentDistance: float = abs(cluster['y_corrected']) -ACTUAL_RADAR_OFFSET
                    currentVelocity: float = abs(cluster['radial_velocity'])
                    stoppingDistance: float = abs(calcStoppingDistance(currentVelocity))
                    print(f"Distance: {currentDistance:.2f}, adjusted with offset {ACTUAL_RADAR_OFFSET:.1f} m")

                    if stoppingDistance >= currentDistance:
                        alarm_tracker.trigger(
                            2,
                            f"Detected vehicle won't be able to stop in time {stoppingDistance:.1f}m > {currentDistance:.1f}m",
                            0.0,
                            currentTime,
                        )
                    elif currentVelocity > SPEED_LIMIT:
                        alarm_tracker.trigger(
                            1,
                            f"Speed limit exceeded by cluster: {currentVelocity:.1f} m/s",
                            0.0,
                            currentTime,
                        )

                if debug_visuals:
                    pass

            results = model.track(
                source=frame,
                imgsz=IMGSZ,
                conf=CONF_THRESHOLD,
                persist=True,
                verbose=False,
                device=0 if device == 'cuda' else 'cpu',
                tracker='bytetrack.yaml',
                classes=ALLOWED_CLASSES_IDS,
            )
            annotatedFrame = frame.copy()

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
                        correctionFunc,
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

                    if car_csv_path:
                        save_car_to_csv(car, trackId, frameIndex, car_csv_path)

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

                                if frameIndex - laneDepartureCooldown.get(
                                    trackId,
                                    -LANE_DEPARTURE_COOLDOWN_FRAMES,
                                ) >= LANE_DEPARTURE_COOLDOWN_FRAMES:
                                    alarm_tracker.trigger(
                                        1,
                                        f"Lane departure car {trackId} crossed {direction} line!",
                                        0.0,
                                        currentTime,
                                    )
                                    laneDepartureCooldown[trackId] = frameIndex
                    # --- END LANE DEPARTURE WATCHDOG ---

                    # points = np.array(car.history).astype(np.int32).reshape((-1, 1, 2))
                    # cv2.polylines(annotatedFrame, [points], False, TRACK_COLOR, LINE_THICKNESS)

                # OVERTAKING

                # determining overtaking zone line position
                if precomputed_overtaking_line_y is None and overtakingLineTriggered and overtakingLineY is None:
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
                            alarm_tracker.trigger(1, 'overtake', 0.0, currentTime)
                            overtakeCooldown[pair] = frameIndex

                prevZoneRanking = currentZoneRanking

            # overtaking line visualization
            if overtakingLineY is not None:
                cv2.line(annotatedFrame, (0, overtakingLineY), (frameWidth, overtakingLineY), (0, 255, 255), 2)

            cv2.putText(
                annotatedFrame,
                f"Frame: {frameIndex}",
                (TEXT_POSITION_X, TEXT_POSITION_Y_START),
                cv2.FONT_HERSHEY_SIMPLEX,
                TEXT_SCALE,
                TEXT_COLOR,
                TEXT_THICKNESS,
            )
            cv2.putText(
                annotatedFrame,
                f"Time: {currentTime:.2f}s",
                (TEXT_POSITION_X, TEXT_POSITION_Y_START + TEXT_LINE_SPACING),
                cv2.FONT_HERSHEY_SIMPLEX,
                TEXT_SCALE,
                TEXT_COLOR,
                TEXT_THICKNESS,
            )

            alarm_tracker.draw(annotatedFrame, currentTime)
            if out is not None:
                out.write(annotatedFrame)

            if show_window:
                previewFrame = cv2.resize(
                    annotatedFrame,
                    None,
                    fx=DISPLAY_SCALE,
                    fy=DISPLAY_SCALE,
                    interpolation=cv2.INTER_AREA,
                )
                cv2.imshow(WINDOW_NAME, previewFrame)
                if cv2.waitKey(WAIT_KEY_MS) & 0xFF == EXIT_KEY:
                    break

            frameIndex += 1

    except Exception as e:
        print(f"Error: {e}")
    finally:
        cap.release()
        if out is not None:
            out.release()
        if show_window:
            cv2.destroyAllWindows()

    return (
        alarm_tracker.active,
        alarm_tracker.format_reasons(),
        lines_dick,
        road_width_h0_px,
        overtakingLineY,
        baseDepthMap,
    )


def run_batch(model: YOLO, device: str) -> None:
    recordings = discover_batch_recordings(DATASET_ROOT, csv_suffix=BATCH_CSV_SUFFIX)
    if not recordings:
        print(f"[INFO] No valid MP4/CSV pairs found in: {DATASET_ROOT}")
        return

    if BATCH_MAX_RECORDINGS is not None:
        recordings = recordings[:BATCH_MAX_RECORDINGS]

    print(f"[INFO] Recordings to analyze: {len(recordings)}")

    lat = 54.37163
    lon = 18.61898
    shared_weather = getWeather(lat, lon, date.today(), 12)
    report_rows: List[Dict[str, str]] = []
    correct_count = 0
    shared_lines: Optional[dict] = None
    shared_road_width_px: Optional[float] = None
    shared_overtaking_line_y: Optional[int] = None
    shared_depth_map: Optional[np.ndarray] = None

    for index, (folder_label, video_path, csv_path) in enumerate(recordings, start=1):
        print("\n" + "=" * 90)
        print(f"[BATCH] {index}/{len(recordings)} | folder={folder_label} | video={video_path}")

        output_video_path = None
        if BATCH_WRITE_VIDEOS:
            output_video_path = build_output_video_path(
                video_path=video_path,
                base_output_dir=os.path.join(BATCH_OUTPUT_DIR, folder_label),
                prefix=f"{index:03d}",
            )

        car_csv_path = os.path.join(
            BATCH_OUTPUT_DIR,
            folder_label,
            f"{index:03d}_{Path(video_path).stem}_cars.csv",
        )
        lines_json_path = os.path.join(
            BATCH_OUTPUT_DIR,
            "lines",
            f"{index:03d}_{Path(video_path).stem}.json",
        )
        depth_output_dir = os.path.join(
            BATCH_OUTPUT_DIR,
            "depth_maps",
            folder_label,
            f"{index:03d}_{Path(video_path).stem}",
        )
        npy_path = os.path.join(depth_output_dir, "base_depth.npy")

        alarm_detected, alarm_reasons, detected_lines, road_width_h0_px, overtaking_line_y, depth_map = run_single_recording(
            model=model,
            device=device,
            video_path=video_path,
            radar_csv_path=csv_path,
            output_video_path=output_video_path,
            car_csv_path=car_csv_path,
            lines_json_path=lines_json_path,
            depth_output_dir=depth_output_dir,
            npy_path=npy_path,
            show_window=BATCH_SHOW_WINDOW,
            write_output=BATCH_WRITE_VIDEOS,
            debug_visuals=BATCH_DEBUG_VISUALS,
            precomputed_lines=shared_lines,
            precomputed_road_width_px=shared_road_width_px,
            precomputed_overtaking_line_y=shared_overtaking_line_y,
            precomputed_depth_map=shared_depth_map,
            weather=shared_weather,
        )

        if shared_lines is None and detected_lines is not None:
            shared_lines = detected_lines
        if shared_road_width_px is None and road_width_h0_px is not None:
            shared_road_width_px = road_width_h0_px
        if shared_overtaking_line_y is None and overtaking_line_y is not None:
            shared_overtaking_line_y = overtaking_line_y
        if shared_depth_map is None and depth_map is not None:
            shared_depth_map = depth_map

        expected_codes = parse_expected_codes_from_path(video_path)
        detected_codes = parse_detected_codes_from_reasons(alarm_reasons)
        expected_alarm = bool(expected_codes)
        codes_match = set(expected_codes) == set(detected_codes)
        if codes_match:
            correct_count += 1

        print(
            f"[BATCH][RESULT] alarmDetected={alarm_detected} | reasons={alarm_reasons} | "
            f"expectedCodes={format_codes(expected_codes)} | detectedCodes={format_codes(detected_codes)} "
            f"| match={codes_match}"
        )

        report_rows.append(
            {
                "expected_folder": folder_label,
                "video_path": os.path.relpath(video_path, DATASET_ROOT),
                "csv_path": os.path.relpath(csv_path, DATASET_ROOT),
                "alarm_detected": str(alarm_detected),
                "alarm_reasons": alarm_reasons,
                "expected_codes": format_codes(expected_codes),
                "detected_codes": format_codes(detected_codes),
                "expected_alarm": str(expected_alarm),
                "match": str(codes_match),
            }
        )

    report_path = build_batch_report_path(REPORTS_DIR, date.today())
    save_batch_report_markdown(report_rows, report_path, correct_count)

    total = len(report_rows)
    accuracy = (correct_count / total) * 100.0 if total else 0.0
    print("\n" + "=" * 90)
    print(f"[BATCH][SUMMARY] Correct: {correct_count}/{total} ({accuracy:.2f}%)")
    print(f"[BATCH][SUMMARY] Report saved to: {report_path}")


if __name__ == "__main__":
    model = YOLO(YOLO_MODEL_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if RUN_MODE == "batch":
        run_batch(model, device)
    else:
        run_single_recording(
            model=model,
            device=device,
            video_path=VIDEO_PATH,
            radar_csv_path=RADAR_CSV_PATH,
            output_video_path=OUTPUT_VIDEO_PATH,
            car_csv_path=CSV_PATH,
            lines_json_path=LINES_JSON_PATH,
            depth_output_dir=DEPTH_OUTPUT_DIR,
            npy_path=NPY_PATH,
            show_window=True,
            write_output=True,
            debug_visuals=True,
        )


