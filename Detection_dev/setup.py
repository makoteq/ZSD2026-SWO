"""
Initializes the processing pipeline and generates the runtime configuration.

The script loads video and radar parameters, estimates lane geometry,
retrieves environmental data, generates a reference depth map, and
stores the resulting configuration for subsequent processing.
"""
import os
import json
import torch
import cv2
from ultralytics import YOLO
from datetime import date

# Local imports
from algorithms.lane_detection.lane_detector import LaneDetector
from utils.radar import Radar
from utils.depth_v2 import computeDepthMap, removeVehiclesFromDepthMap, saveDepthMap
from utils.weather import getWeather

CURRENT_SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
SCENARIO = "output"
DATA_DIR = os.path.abspath(os.path.join(CURRENT_SCRIPT_PATH, "..", "data"))

#path to control data
VIDEO_PATH = os.path.join(DATA_DIR, "dataset/noalarm/1_Control/1_Control.mp4")
RADAR_CSV_PATH = os.path.join(DATA_DIR, "dataset/noalarm/1_Control/1_Control--6.0.csv")

YOLO_MODEL_PATH = os.path.join(DATA_DIR, "models", "best.pt")
DEPTH_MODEL_PATH = os.path.join(DATA_DIR, "models", "depth_anything_v2_vits.pth")
DEPTH_LIB_PATH = os.path.join(DATA_DIR, "models", "Depth-Anything-V2")
DEPTH_OUTPUT_DIR = os.path.join(DATA_DIR, "output", "depth_maps")
NPY_PATH = os.path.join(DATA_DIR, "config", "base_depth.npy")
CONFIG_JSON_PATH = os.path.join(DATA_DIR, "config", "config.json")

os.makedirs(os.path.join(DATA_DIR, "output"), exist_ok=True)
os.makedirs(os.path.join(DATA_DIR, "config"), exist_ok=True)
os.makedirs(DEPTH_OUTPUT_DIR, exist_ok=True)

# yolo
ROAD_WIDTH_METERS = 7.0
FOV = 20.0

START_TIME = 0.0
CONF_THRESHOLD = 0.8
IMGSZ = 800
ALLOWED_CLASSES_IDS = [0]

# radar
MASK_Z_MIN = 30.0
MASK_Z_MAX = 100.0
MASK_Y_MIN = 0.0
MASK_Y_MAX = 120.0

LAT = 54.37163
LON = 18.61898

def generate_configuration():
    """
    Generate the runtime configuration from video, radar,
    weather, lane detection, and depth estimation data.
    """

    # correctionFunc = plotYOffsetCorrelation(CSV_PATH)

    correctionFunc = lambda x: 0.0
    model = YOLO(YOLO_MODEL_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    radar_delay = 0.0

    with open(RADAR_CSV_PATH, 'r') as f:
        f.readline()
        first_line = f.readline().split(',')
        radar_delay = float(first_line[0])
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        exit(1)
    radar: Radar = Radar(RADAR_CSV_PATH, START_TIME + radar_delay)
    radar.applyMask(MASK_Z_MIN, MASK_Z_MAX, MASK_Y_MIN, MASK_Y_MAX)
    radar.addNoise()
    radar.findLane()
    radar.visualize()

    fps = cap.get(cv2.CAP_PROP_FPS)

    Weather = getWeather(LAT, LON, date.today(), 12)

    frame_time = 1.0 / fps
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(START_TIME * fps))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Depth map for the first frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(START_TIME * fps))
    _, firstFrame = cap.read()
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(START_TIME * fps))
 
    if not os.path.exists(NPY_PATH):
        rawDepthMap = computeDepthMap(NPY_PATH, firstFrame, DEPTH_MODEL_PATH, DEPTH_LIB_PATH, DEPTH_OUTPUT_DIR)
        firstFrameResults = model.predict(source=firstFrame, imgsz=IMGSZ, conf=CONF_THRESHOLD, verbose=False,
                                          device=0 if device == 'cuda' else 'cpu', classes=ALLOWED_CLASSES_IDS)
        firstFrameBboxes = [
            {'x1': box[0], 'y1': box[1], 'x2': box[2], 'y2': box[3]}
            for box in firstFrameResults[0].boxes.xyxy.cpu().numpy()
        ] if firstFrameResults[0].boxes is not None else []
        baseDepthMap = removeVehiclesFromDepthMap(rawDepthMap, firstFrameBboxes, paddingFactor=0.05)
        saveDepthMap(baseDepthMap, os.path.join(DATA_DIR, "config"), DEPTH_OUTPUT_DIR, name="base_depth")
    else:
        print("DepthV2: depth map already exists, skipping.")


    detector = LaneDetector()
    lines_dick = detector.detect(firstFrame)
    xLeft = lines_dick["left_line"]["start"][0]
    xRight = lines_dick["right_line"]["start"][0]
    road_width_h0_px = abs(xRight - xLeft)

    # we are using only left and right lines
    detected_lines = [lines_dick["left_line"],lines_dick["right_line"]]  # converting dictionary into array for easier use

    depthMeta = {
        "npy_path": NPY_PATH,
        "output_dir": DEPTH_OUTPUT_DIR,
        "model_path": DEPTH_MODEL_PATH,
    }

    weatherMeta = {
        "latitude": LAT,
        "longitude": LON,
        "date": str(date.today()),
        "condition": Weather.condition,
        "description": Weather.description,
        "Markiplier": Weather.multiplier,
    }

    config = {
        "geometry": {
            "road_width_meters": ROAD_WIDTH_METERS,
            "road_width_h0_px": road_width_h0_px,
            "fov_deg": FOV,
        },
        "lanes": lines_dick,
        "depth": depthMeta,
        "weather": weatherMeta,
    }

    with open(CONFIG_JSON_PATH, 'w', encoding='utf-8') as f:
        json.dump(config, f, indent=4, ensure_ascii=False)

if __name__ == "__main__":
    generate_configuration()