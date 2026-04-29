import os
import torch
import cv2
import numpy as np
from ultralytics import YOLO
from keras import models
from pathlib import Path
from typing import Dict, Final, List, Tuple

# Importy lokalne
from utils.car import Car
from utils.radar import Radar
from utils.utilsBATCH import drawCustomBox, plotRadarComparison, matchClustersToCars, getManualLaneLines
from utils.depth_v2 import DepthV2, rankCarsByDepth


CURRENT_SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(CURRENT_SCRIPT_PATH, "..", "data"))

# Control
# VIDEO_PATH = os.path.join(DATA_DIR, "normal_traffic/lines_det2.mp4")
# CSV_PATH = os.path.join(DATA_DIR, "alarm/speeding1/radar_points_world.csv")

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

VIDEO_PATH = os.path.join(DATA_DIR, "dataset/noalarm/1_control.mp4")
CSV_PATH = os.path.join(DATA_DIR, "dataset/noalarm/1_control.csv")

YOLO_MODEL_PATH = os.path.join(DATA_DIR, "models", "best.pt")
CNN_MODEL_PATH = os.path.join(DATA_DIR, "models", "cnn.h5")
OUTPUT_VIDEO_PATH = os.path.join(DATA_DIR, "output", "trajectory.mp4")
DEPTH_MODEL_PATH = os.path.join(DATA_DIR, "models", "depth_anything_v2_vits.pth")
DEPTH_LIB_PATH = os.path.join(DATA_DIR, "models", "Depth-Anything-V2")
DEPTH_OUTPUT_DIR = os.path.join(DATA_DIR, "output")
DATASET_DIR = os.path.join(DATA_DIR, "dataset")
BATCH_REPORT_PATH = os.path.join(DEPTH_OUTPUT_DIR, "batch_alarm_report.md")
BATCH_OUTPUT_DIR = os.path.join(DEPTH_OUTPUT_DIR, "batch")

# Run config
RUN_MODE = "single"  # "single" or "batch"
RUN_MODE_DEBUG = True
SHOW_WINDOW_IN_SINGLE = True

# yolo
ROAD_WIDTH_METERS = 7.0
FOV = 14.0

START_TIME = 0
CONF_THRESHOLD = 0.7
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

LANE_DEPARTURE_COLOR: Final[tuple] = (0, 0, 255)
ALARM_COLOR: Final[tuple] = (0, 0, 255)
ALARM_SQUARE_SIZE: Final[int] = 40
ALARM_MARGIN: Final[int] = 20
ALARM_ACTIVE = False
ALARM_REASON_COUNTS: Dict[str, int] = {}

BOX_COLOR: Final[tuple] = (0, 255, 0)
BOX_THICKNESS: Final[int] = 2

SPEED_LIMIT_KMH: Final[float] = 50.0
SPEED_LIMIT: Final[float] = SPEED_LIMIT_KMH / 3.6
OVERTAKING_DEPTH_MARGIN: Final[float] = 0.0
# radar
RADAR_STEP_INTERVAL = 10
MASK_Z_MIN = 30.0
MASK_Z_MAX = 50.0
MASK_Y_MIN = 52.0
MASK_Y_MAX = 200.0
RADAR_SYNC_MODE = "auto_end"  # "manual", "auto_start", "auto_end", "auto_center"
RADAR_TIME_OFFSET_SEC = 0.0

# window
WINDOW_NAME = "Traffic Analysis"
WAIT_KEY_MS = 1
EXIT_KEY = ord('q')

def activateAlarm(reason: str = "unknown") -> None:
    global ALARM_ACTIVE, ALARM_REASON_COUNTS
    ALARM_ACTIVE = True
    ALARM_REASON_COUNTS[reason] = ALARM_REASON_COUNTS.get(reason, 0) + 1
    print("]ALARM!]")


def formatAlarmReasons() -> str:
    if not ALARM_REASON_COUNTS:
        return "none"

    parts: List[str] = []
    for reason in sorted(ALARM_REASON_COUNTS.keys()):
        parts.append(f"{reason}({ALARM_REASON_COUNTS[reason]})")
    return "; ".join(parts)


def detectOvertakesFromRanking(
    previousOrder: Dict[int, int],
    currentRanking: List[dict],
    depthMargin: float,
) -> Tuple[List[Tuple[int, int]], Dict[int, int]]:
    currentOrder = {int(item['id']): idx for idx, item in enumerate(currentRanking)}
    currentDepth = {int(item['id']): float(item['depth']) for item in currentRanking}
    overtakes: List[Tuple[int, int]] = []

    if not previousOrder:
        return overtakes, currentOrder

    commonIds = [carId for carId in currentOrder.keys() if carId in previousOrder]

    for idx, firstId in enumerate(commonIds):
        for secondId in commonIds[idx + 1:]:
            firstPrev = previousOrder[firstId]
            secondPrev = previousOrder[secondId]
            firstCurrent = currentOrder[firstId]
            secondCurrent = currentOrder[secondId]
            currentDepthDiff = abs(currentDepth[firstId] - currentDepth[secondId])

            # Overtake happens when order between two cars is reversed.
            if firstPrev > secondPrev and firstCurrent < secondCurrent and currentDepthDiff >= depthMargin:
                overtakes.append((firstId, secondId))
            elif secondPrev > firstPrev and secondCurrent < firstCurrent and currentDepthDiff >= depthMargin:
                overtakes.append((secondId, firstId))

    return overtakes, currentOrder


def computeRadarStartTime(
    radarTimeMin: float,
    radarTimeMax: float,
    videoDuration: float,
    baseStartTime: float,
    syncMode: str,
    timeOffsetSec: float,
) -> float:
    mode = syncMode.lower().strip()

    if mode == "manual":
        return max(radarTimeMin, baseStartTime + timeOffsetSec)

    if mode == "auto_start":
        return max(radarTimeMin, radarTimeMin + timeOffsetSec)

    if mode == "auto_center":
        centerStart = radarTimeMin + max(0.0, (radarTimeMax - radarTimeMin - videoDuration) / 2.0)
        return max(radarTimeMin, centerStart + timeOffsetSec)

    # default: auto_end
    endAlignedStart = radarTimeMax - videoDuration
    return max(radarTimeMin, endAlignedStart + timeOffsetSec)


def discoverBatchRecordings(datasetRoot: str) -> List[Tuple[str, str, str]]:
    recordings: List[Tuple[str, str, str]] = []

    for folderLabel in ("alarm", "noalarm"):
        folderPath = os.path.join(datasetRoot, folderLabel)
        if not os.path.isdir(folderPath):
            print(f"[INFO] Brak folderu: {folderPath}")
            continue

        for currentRoot, _, files in os.walk(folderPath):
            mp4Files = sorted([f for f in files if f.lower().endswith(".mp4")])
            for mp4Name in mp4Files:
                videoPath = os.path.join(currentRoot, mp4Name)
                csvName = f"{Path(mp4Name).stem}.csv"
                csvPath = os.path.join(currentRoot, csvName)

                if not os.path.isfile(csvPath):
                    print(f"[WARNING] Pomijam plik bez CSV: {videoPath} (oczekiwany: {csvPath})")
                    continue

                recordings.append((folderLabel, videoPath, csvPath))

    return recordings


def saveBatchReportMarkdown(reportRows: List[Dict[str, str]], outputPath: str, correctCount: int) -> None:
    os.makedirs(os.path.dirname(outputPath), exist_ok=True)
    total = len(reportRows)
    accuracy = (correctCount / total) * 100.0 if total else 0.0

    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for row in reportRows:
        expectedAlarm = row["expected_alarm"] == "True"
        detectedAlarm = row["alarm_detected"] == "True"

        if expectedAlarm and detectedAlarm:
            tp += 1
        elif not expectedAlarm and not detectedAlarm:
            tn += 1
        elif not expectedAlarm and detectedAlarm:
            fp += 1
        else:
            fn += 1

    falseAlarmRate = (fp / (fp + tn) * 100.0) if (fp + tn) else 0.0
    trueAlarmRate = (tp / (tp + fn) * 100.0) if (tp + fn) else 0.0

    def esc(value: str) -> str:
        return value.replace("|", "\\|")

    mdLines: List[str] = [
        "# Batch Report",
        "",
        f"- Quantity of recordings: **{total}**",
        f"- Correct classifications: **{correctCount}**",
        f"- Accuracy: **{accuracy:.2f}%**",
        f"- False alarm rate (FAR): **{falseAlarmRate:.2f}%**",
        f"- True alarm rate (TAR): **{trueAlarmRate:.2f}%**",
        "",
        "| # | expected_folder | alarm_detected | alarm_reasons | expected_alarm | match | video_path | csv_path |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]

    for idx, row in enumerate(reportRows, start=1):
        mdLines.append(
            f"| {idx} | {esc(row['expected_folder'])} | {esc(row['alarm_detected'])} | "
            f"{esc(row['alarm_reasons'])} | {esc(row['expected_alarm'])} | {esc(row['match'])} | {esc(row['video_path'])} | {esc(row['csv_path'])} |"
        )

    with open(outputPath, "w", encoding="utf-8") as reportFile:
        reportFile.write("\n".join(mdLines) + "\n")


def buildOutputVideoPath(videoPath: str, baseOutputDir: str, prefix: str = "") -> str:
    stem = Path(videoPath).stem
    safePrefix = f"{prefix}_" if prefix else ""
    fileName = f"{safePrefix}{stem}_annotated.mp4"
    os.makedirs(baseOutputDir, exist_ok=True)
    return os.path.join(baseOutputDir, fileName)


def runSingleRecording(
    model: YOLO,
    cnn,
    device: str,
    videoPath: str,
    csvPath: str,
    outputVideoPath: str,
    showWindow: bool,
) -> Tuple[bool, str]:
    global ALARM_ACTIVE, ALARM_REASON_COUNTS
    ALARM_ACTIVE = False
    ALARM_REASON_COUNTS = {}

    if not os.path.isfile(videoPath):
        raise FileNotFoundError(f"Nie znaleziono nagrania: {videoPath}")

    if not os.path.isfile(csvPath):
        raise FileNotFoundError(f"Nie znaleziono pliku CSV: {csvPath}")

    showPreview = showWindow and RUN_MODE_DEBUG

    cap = cv2.VideoCapture(videoPath)
    if not cap.isOpened():
        raise RuntimeError(f"Nie udalo sie otworzyc nagrania: {videoPath}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0:
        cap.release()
        raise RuntimeError(f"Niepoprawne FPS dla pliku: {videoPath}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    videoDuration = total_frames / fps

    radarProbe: Radar = Radar(csvPath, 0.0)
    radarTimeMin = float(radarProbe.dataFrame["timestamp"].min()) if not radarProbe.dataFrame.empty else 0.0
    radarTimeMax = float(radarProbe.dataFrame["timestamp"].max()) if not radarProbe.dataFrame.empty else 0.0
    radarStartTime = computeRadarStartTime(
        radarTimeMin=radarTimeMin,
        radarTimeMax=radarTimeMax,
        videoDuration=videoDuration,
        baseStartTime=START_TIME,
        syncMode=RADAR_SYNC_MODE,
        timeOffsetSec=RADAR_TIME_OFFSET_SEC,
    )
    print(
        f"[SYNC] videoDuration={videoDuration:.2f}s | radarRange=({radarTimeMin:.2f}, {radarTimeMax:.2f})s "
        f"| mode={RADAR_SYNC_MODE} | radarStart={radarStartTime:.2f}s"
    )

    radar: Radar = Radar(csvPath, radarStartTime)
    autoMaskYMin = MASK_Y_MIN
    autoMaskYMax = MASK_Y_MAX
    if not radar.dataFrame.empty:
        autoMaskYMin = float(radar.dataFrame["y_corrected"].min())
        autoMaskYMax = float(radar.dataFrame["y_corrected"].max())

    radar.applyMask(MASK_Z_MIN, MASK_Z_MAX, MASK_Y_MIN, MASK_Y_MAX)
    if radar.dataFrame.empty:
        print(
            f"[WARNING] Radar mask Y=({MASK_Y_MIN}, {MASK_Y_MAX}) wyciela wszystkie punkty dla: {csvPath}. "
            f"Fallback do zakresu danych Y=({autoMaskYMin:.2f}, {autoMaskYMax:.2f})."
        )
        radar = Radar(csvPath, radarStartTime)
        radar.applyMask(MASK_Z_MIN, MASK_Z_MAX, autoMaskYMin, autoMaskYMax)

    if radar.dataFrame.empty:
        raise RuntimeError(
            f"Brak punktow radarowych po maskowaniu dla pliku: {csvPath}. "
            "Sprawdz zakresy maski i kolumny CSV."
        )

    radar.findLane()

    print(f"FPS: {fps}")
    frame_time = 1.0 / fps
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(START_TIME * fps))
    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    os.makedirs(os.path.dirname(outputVideoPath), exist_ok=True)
    out = cv2.VideoWriter(outputVideoPath, fourcc, int(fps), (frameWidth, frameHeight))

    depthProcessor = DepthV2(modelPath=DEPTH_MODEL_PATH, libPath=DEPTH_LIB_PATH)

    cap.set(cv2.CAP_PROP_POS_FRAMES, total_frames - 1)
    successLastFrame, lastFrame = cap.read()
    if not successLastFrame:
        cap.release()
        out.release()
        raise RuntimeError(f"Nie udalo sie odczytac ostatniej klatki: {videoPath}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, int(START_TIME * fps))

    baseDepthMap = depthProcessor.getDepthMap(lastFrame)
    depthProcessor.saveDepthMap(baseDepthMap, DEPTH_OUTPUT_DIR, name="base_depth")

    carsDict: Dict[int, Car] = {}
    previousDepthOrder: Dict[int, int] = {}
    frameIndex = 0
    posGlobalYDifference = 0.0

    try:
        while cap.isOpened():

            success, frame = cap.read()
            if not success:
                break

            staleIds = [carId for carId, carObj in carsDict.items() if carObj.lastSeen < frameIndex - 5]
            for carId in staleIds:
                del carsDict[carId]

            if frameIndex == 0:
                lines = getManualLaneLines(VIDEO_PATH)
                #detected_lines = [{'m': -0.6913385826771653, 'b': 876.2346456692912, 'x_bot': -451.13543307086616, 'abs_m': 0.6913385826771653}, {'m': 0.532235939643347, 'b': 268.9657064471879, 'x_bot': 1290.8587105624142, 'abs_m': 0.532235939643347}]

                y = 0
                xLeft = (lines[0]["m"] * y) + lines[0]["b"]
                xRight = (lines[1]["m"] * y) + lines[1]["b"]
                road_width_h0_px = abs(xRight - xLeft)

            if frameIndex % RADAR_STEP_INTERVAL == 0:
                radar_setp = frame_time * RADAR_STEP_INTERVAL
                radar.step(radar_setp)
                radar.clusterPoints()
                clusterCenters = radar.getClusterCenters()

                for cluster in clusterCenters:
                    currentVelocity: float = abs(cluster["radial_velocity"])
                    if currentVelocity > SPEED_LIMIT:
                        activateAlarm("speed_limit_exceeded")
                        print(f"[WARNING] Speed limit exceeded by cluster: {currentVelocity:.2f} m/s")

                if RUN_MODE_DEBUG:
                    plotRadarComparison(radar.minX, radar.maxX, 0, radar.maxY, carsDict, clusterCenters)
                matchedDist = matchClustersToCars(carsDict, clusterCenters, frameIndex)
                if posGlobalYDifference == 0.0 and matchedDist > 0.0:
                    posGlobalYDifference = matchedDist

            results = model.track(
                source=frame,
                imgsz=IMGSZ,
                conf=CONF_THRESHOLD,
                persist=True,
                verbose=False,
                device=0 if device == "cuda" else "cpu",
                tracker="bytetrack.yaml",
                classes=ALLOWED_CLASSES_IDS,
            )
            annotatedFrame = frame.copy()

            LANE_LINE_COLOR = (0, 200, 255)
            LANE_LINE_THICKNESS = 2
            y_top = 0
            y_bottom = frameHeight - 1
            for line in lines:
                m = line.get("m")
                b = line.get("b")
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

                currentFrameCars = [
                    {
                        "id": tid,
                        "x1": int(box[0]),
                        "y1": int(box[1]),
                        "x2": int(box[2]),
                        "y2": int(box[3]),
                    }
                    for tid, box in zip(trackIds, boxesXyxy)
                ]
                ranked = rankCarsByDepth(baseDepthMap, currentFrameCars)
                print(f"Frame {frameIndex} | ranking: {ranked}")
                overtakes, previousDepthOrder = detectOvertakesFromRanking(
                    previousDepthOrder,
                    ranked,
                    OVERTAKING_DEPTH_MARGIN,
                )
                for overtakerId, overtakenId in overtakes:
                    activateAlarm("overtaking_detected")
                    print(
                        f"[WARNING] Overtaking detected by depth ranking | overtaker ID={overtakerId} "
                        f"passed ID={overtakenId} | frame={frameIndex}"
                    )

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
                        lines,
                        road_width_h0_px,
                        FOV,
                        frame_time,
                        IMGSZ,
                        radar,
                        lambda _dist: posGlobalYDifference,
                    )

                    laneDepartureDetected = car.updateLaneState(
                        lines,
                        frameIndex,
                        centerX=float(boxXywh[0]),
                        centerY=float(boxXywh[1]),
                    )
                    if laneDepartureDetected:
                        activateAlarm("lane_departure")
                        print(f"[WARNING] Lane departure | ID={trackId} | frame={frameIndex}")

                    if car.isOutsideLane:
                        print(f"[WARNING] ID={trackId} | frame={frameIndex}")

                    drawCustomBox(
                        annotatedFrame,
                        boxXyxy,
                        trackId,
                        conf,
                        car.type,
                        car.pos[-1].x,
                        car.pos[-1].y,
                        car.size[-1].w,
                        car.size[-1].h,
                        car.velo[-1].v,
                    )

                    points = np.array(car.history).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotatedFrame, [points], False, TRACK_COLOR, LINE_THICKNESS)

            currentTime = START_TIME + (frameIndex * frame_time)
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

            if ALARM_ACTIVE:
                x2 = frameWidth - ALARM_MARGIN
                y1 = ALARM_MARGIN
                x1 = x2 - ALARM_SQUARE_SIZE
                y2 = y1 + ALARM_SQUARE_SIZE
                cv2.rectangle(annotatedFrame, (x1, y1), (x2, y2), ALARM_COLOR, -1)

            out.write(annotatedFrame)
            if showPreview:
                cv2.imshow(WINDOW_NAME, annotatedFrame)
                if cv2.waitKey(WAIT_KEY_MS) & 0xFF == EXIT_KEY:
                    break

            frameIndex += 1

    except Exception as e:
        print(f"Blad: {e}")
    finally:
        cap.release()
        out.release()
        if showPreview:
            cv2.destroyAllWindows()

    return ALARM_ACTIVE, formatAlarmReasons()


def runBatch(model: YOLO, cnn, device: str, datasetRoot: str) -> None:
    recordings = discoverBatchRecordings(datasetRoot)
    if not recordings:
        print(f"[INFO] Nie znaleziono poprawnych par MP4/CSV w: {datasetRoot}")
        return

    print(f"[INFO] Liczba nagran do analizy: {len(recordings)}")

    reportRows: List[Dict[str, str]] = []
    correctCount = 0

    for index, (folderLabel, videoPath, csvPath) in enumerate(recordings, start=1):
        print("\n" + "=" * 90)
        print(f"[BATCH] {index}/{len(recordings)} | folder={folderLabel} | video={videoPath}")

        outputVideoPath = buildOutputVideoPath(
            videoPath=videoPath,
            baseOutputDir=os.path.join(BATCH_OUTPUT_DIR, folderLabel),
            prefix=f"{index:03d}",
        )

        alarmDetected, alarmReasons = runSingleRecording(
            model=model,
            cnn=cnn,
            device=device,
            videoPath=videoPath,
            csvPath=csvPath,
            outputVideoPath=outputVideoPath,
            showWindow=RUN_MODE_DEBUG,
        )

        expectedAlarm = folderLabel == "alarm"
        isMatch = alarmDetected == expectedAlarm
        if isMatch:
            correctCount += 1

        print(
            f"[BATCH][RESULT] alarmDetected={alarmDetected} | reasons={alarmReasons} | expected={expectedAlarm} "
            f"| match={isMatch}"
        )

        reportRows.append(
            {
                "expected_folder": folderLabel,
                "video_path": os.path.relpath(videoPath, datasetRoot),
                "csv_path": os.path.relpath(csvPath, datasetRoot),
                "alarm_detected": str(alarmDetected),
                "alarm_reasons": alarmReasons,
                "expected_alarm": str(expectedAlarm),
                "match": str(isMatch),
            }
        )

    saveBatchReportMarkdown(reportRows, BATCH_REPORT_PATH, correctCount)

    total = len(reportRows)
    accuracy = (correctCount / total) * 100.0 if total else 0.0
    print("\n" + "=" * 90)
    print(f"[BATCH][SUMMARY] Correct: {correctCount}/{total} ({accuracy:.2f}%)")
    print(f"[BATCH][SUMMARY] Raport zapisany do: {BATCH_REPORT_PATH}")

if __name__ == "__main__":
    model = YOLO(YOLO_MODEL_PATH)
    cnn = models.load_model(CNN_MODEL_PATH, compile=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if RUN_MODE == "single":
        alarmDetected, alarmReasons = runSingleRecording(
            model=model,
            cnn=cnn,
            device=device,
            videoPath=VIDEO_PATH,
            csvPath=CSV_PATH,
            outputVideoPath=OUTPUT_VIDEO_PATH,
            showWindow=SHOW_WINDOW_IN_SINGLE,
        )
        print(f"[SINGLE][RESULT] alarmDetected={alarmDetected} | reasons={alarmReasons}")
    elif RUN_MODE == "batch":
        runBatch(
            model=model,
            cnn=cnn,
            device=device,
            datasetRoot=DATASET_DIR,
        )
    else:
        raise ValueError(f"Nieznany RUN_MODE={RUN_MODE}. Uzyj: 'single' albo 'batch'.")

