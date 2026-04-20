import os
import json
from collections import defaultdict

import cv2
import numpy as np
import torch
from ultralytics import YOLO

CURRENT_SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(CURRENT_SCRIPT_PATH, "../../..", "data"))
YOLO_MODEL_PATH = os.path.join(DATA_DIR, "models", "best.pt")
OUTPUT_VIDEO_PATH = os.path.join(DATA_DIR, "output", "trajectory.mp4")
LINES_JSON_PATH = os.path.join(DATA_DIR, "output", "lines.json")
DEFAULT_VIDEO_NAME = "rgb.mp4"
START_TIME = 1
CONF_THRESHOLD = 0.5
IMGSZ = 800
ALLOWED_CLASSES_IDS = [0]
MAX_MISSING_FRAMES = 30
VIDEO_FOURCC = 'mp4v'
TRACK_COLOR = (0, 255, 0)
BBOX_COLOR = (0, 255, 0)
LINE_THICKNESS = 1
FINISHED_LINE_THICKNESS = 5
FONT_SCALE = 0.4
FONT_THICKNESS = 1
CORNER_DOT_RADIUS = 3
CORNER_DOT_COLOR = (0, 255, 255)
LINE_LEFT_COLOR = (255, 200, 0)
LINE_RIGHT_COLOR = (0, 200, 255)
LINE_MIDDLE_COLOR = (0, 255, 255)
APPROX_CORNER_LEFT = "left"
APPROX_CORNER_RIGHT = "right"
MIN_POINTS_FOR_LINE_APPROX = 30
# optional bias to make approximated lines more vertical
VERTICAL_BIAS = 0.07

SAVE_LINES_AFTER_PASSES = 10
WINDOW_NAME = "Traffic Analysis"
WAIT_KEY_MS = 1
EXIT_KEY = ord('q')


def drawCustomBox(annotatedFrame: np.ndarray, boxXyxy: np.ndarray, trackId: int, conf: float) -> None:
    x1, y1, x2, y2 = map(int, boxXyxy)
    labelText = f"ID:{trackId} | {conf:.2f}"

    cv2.rectangle(annotatedFrame, (x1, y1), (x2, y2), BBOX_COLOR, LINE_THICKNESS)

    (textWidth, textHeight), _ = cv2.getTextSize(labelText, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS)
    cv2.rectangle(annotatedFrame, (x1, y1 - textHeight - 5), (x1 + textWidth, y1), BBOX_COLOR, -1)
    cv2.putText(annotatedFrame, labelText, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 0), FONT_THICKNESS)


def drawPersistentCornerDots(annotatedFrame: np.ndarray, points: list[tuple[int, int]]) -> None:
    for pointX, pointY in points:
        cv2.circle(annotatedFrame, (pointX, pointY), CORNER_DOT_RADIUS, CORNER_DOT_COLOR, -1)


def approximateLine(
    points: list[tuple[int, int]],
    frameWidth: int,
    frameHeight: int,
    invertVerticalBias: bool = False
) -> tuple[tuple[int, int], tuple[int, int]] | None:
    if len(points) < MIN_POINTS_FOR_LINE_APPROX:
        return None

    pointsArray = np.array(points, dtype=np.float32).reshape(-1, 1, 2)
    vx, vy, x0, y0 = cv2.fitLine(pointsArray, cv2.DIST_L2, 0, 0.01, 0.01)
    vx = float(vx)
    vy = float(vy)
    x0 = float(x0)
    y0 = float(y0)

    verticalVy = 1.0 if vy >= 0 else -1.0
    if invertVerticalBias:
        verticalVy *= -1.0

    vx = (1.0 - VERTICAL_BIAS) * vx
    vy = (1.0 - VERTICAL_BIAS) * vy + VERTICAL_BIAS * verticalVy
    norm = float(np.hypot(vx, vy))
    if norm > 1e-6:
        vx /= norm
        vy /= norm

    intersections: list[tuple[int, int]] = []

    if abs(vx) > 1e-6:
        tAtLeft = (0 - x0) / vx
        yAtLeft = y0 + tAtLeft * vy
        if 0 <= yAtLeft <= frameHeight - 1:
            intersections.append((0, int(round(yAtLeft))))

        tAtRight = (frameWidth - 1 - x0) / vx
        yAtRight = y0 + tAtRight * vy
        if 0 <= yAtRight <= frameHeight - 1:
            intersections.append((frameWidth - 1, int(round(yAtRight))))

    if abs(vy) > 1e-6:
        tAtTop = (0 - y0) / vy
        xAtTop = x0 + tAtTop * vx
        if 0 <= xAtTop <= frameWidth - 1:
            intersections.append((int(round(xAtTop)), 0))

        tAtBottom = (frameHeight - 1 - y0) / vy
        xAtBottom = x0 + tAtBottom * vx
        if 0 <= xAtBottom <= frameWidth - 1:
            intersections.append((int(round(xAtBottom)), frameHeight - 1))

    uniqueIntersections: list[tuple[int, int]] = []
    for point in intersections:
        if point not in uniqueIntersections:
            uniqueIntersections.append(point)

    if len(uniqueIntersections) < 2:
        return None

    if len(uniqueIntersections) == 2:
        return uniqueIntersections[0], uniqueIntersections[1]

    maxDist = -1.0
    bestPair: tuple[tuple[int, int], tuple[int, int]] | None = None
    for idxA in range(len(uniqueIntersections)):
        for idxB in range(idxA + 1, len(uniqueIntersections)):
            pointA = uniqueIntersections[idxA]
            pointB = uniqueIntersections[idxB]
            dist = float(np.hypot(pointA[0] - pointB[0], pointA[1] - pointB[1]))
            if dist > maxDist:
                maxDist = dist
                bestPair = (pointA, pointB)

    return bestPair


def drawFinishedTrackLines(
    annotatedFrame: np.ndarray,
    finishedTrackLinesLeft: dict[int, tuple[tuple[int, int], tuple[int, int]]],
    finishedTrackLinesRight: dict[int, tuple[tuple[int, int], tuple[int, int]]]
) -> None:
    leftFinishedLines: list[tuple[int, tuple[int, int], tuple[int, int]]] = []
    rightFinishedLines: list[tuple[int, tuple[int, int], tuple[int, int]]] = []

    for trackId, (startPoint, endPoint) in finishedTrackLinesLeft.items():
        leftFinishedLines.append((trackId, startPoint, endPoint))
    for trackId, (startPoint, endPoint) in finishedTrackLinesRight.items():
        rightFinishedLines.append((trackId, startPoint, endPoint))

    if not leftFinishedLines and not rightFinishedLines:
        return

    def xNearBottom(lineStart: tuple[int, int], lineEnd: tuple[int, int], frameHeight: int) -> float:
        x1, y1 = lineStart
        x2, y2 = lineEnd

        targetY = frameHeight - 1
        minY = min(y1, y2)
        maxY = max(y1, y2)

        if minY <= targetY <= maxY and y1 != y2:
            t = (targetY - y1) / (y2 - y1)
            return x1 + t * (x2 - x1)

        return float(x1 if y1 > y2 else x2)

    frameHeight = annotatedFrame.shape[0]

    def getExtremeLines() -> tuple[
        tuple[int, tuple[int, int], tuple[int, int]] | None,
        tuple[int, tuple[int, int], tuple[int, int]] | None
    ]:
        leftmostLine = None
        rightmostLine = None

        if leftFinishedLines:
            leftmostLine = min(leftFinishedLines, key=lambda item: xNearBottom(item[1], item[2], frameHeight))
        if rightFinishedLines:
            rightmostLine = max(rightFinishedLines, key=lambda item: xNearBottom(item[1], item[2], frameHeight))

        return leftmostLine, rightmostLine

    leftmostLine, rightmostLine = getExtremeLines()

    def orderedByY(startPoint: tuple[int, int], endPoint: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
        return (startPoint, endPoint) if startPoint[1] <= endPoint[1] else (endPoint, startPoint)

    if leftmostLine is not None:
        trackId, startPoint, endPoint = leftmostLine
        cv2.line(annotatedFrame, startPoint, endPoint, LINE_LEFT_COLOR, FINISHED_LINE_THICKNESS)
        cv2.putText(annotatedFrame, f"L{trackId}", startPoint, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, LINE_LEFT_COLOR, FONT_THICKNESS)

    if rightmostLine is not None:
        trackId, startPoint, endPoint = rightmostLine
        cv2.line(annotatedFrame, startPoint, endPoint, LINE_RIGHT_COLOR, FINISHED_LINE_THICKNESS)
        cv2.putText(annotatedFrame, f"R{trackId}", startPoint, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, LINE_RIGHT_COLOR, FONT_THICKNESS)

    if leftmostLine is not None and rightmostLine is not None:
        _, leftStart, leftEnd = leftmostLine
        _, rightStart, rightEnd = rightmostLine
        leftTop, leftBottom = orderedByY(leftStart, leftEnd)
        rightTop, rightBottom = orderedByY(rightStart, rightEnd)

        middleStart = (
            int(round((leftTop[0] + rightTop[0]) / 2)),
            int(round((leftTop[1] + rightTop[1]) / 2)),
        )
        middleEnd = (
            int(round((leftBottom[0] + rightBottom[0]) / 2)),
            int(round((leftBottom[1] + rightBottom[1]) / 2)),
        )

        cv2.line(annotatedFrame, middleStart, middleEnd, LINE_MIDDLE_COLOR, FINISHED_LINE_THICKNESS)
        cv2.putText(annotatedFrame, "M", middleStart, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, LINE_MIDDLE_COLOR, FONT_THICKNESS)


def saveFinalLines(
    finishedTrackLinesLeft: dict[int, tuple[tuple[int, int], tuple[int, int]]],
    finishedTrackLinesRight: dict[int, tuple[tuple[int, int], tuple[int, int]]],
    frameHeight: int,
    outputPath: str,
    passesThreshold: int,
) -> None:
    def xNearBottom(lineStart: tuple[int, int], lineEnd: tuple[int, int]) -> float:
        x1, y1 = lineStart
        x2, y2 = lineEnd

        targetY = frameHeight - 1
        minY = min(y1, y2)
        maxY = max(y1, y2)

        if minY <= targetY <= maxY and y1 != y2:
            t = (targetY - y1) / (y2 - y1)
            return x1 + t * (x2 - x1)

        return float(x1 if y1 > y2 else x2)

    leftFinal = None
    rightFinal = None

    if finishedTrackLinesLeft:
        leftFinal = min(
            finishedTrackLinesLeft.items(),
            key=lambda item: xNearBottom(item[1][0], item[1][1])
        )
    if finishedTrackLinesRight:
        rightFinal = max(
            finishedTrackLinesRight.items(),
            key=lambda item: xNearBottom(item[1][0], item[1][1])
        )

    payload = {
        "passes_threshold": passesThreshold,
        "vertical_bias": VERTICAL_BIAS,
        "left_line": None,
        "right_line": None,
        "middle_line": None,
    }

    def orderedByY(startPoint: tuple[int, int], endPoint: tuple[int, int]) -> tuple[tuple[int, int], tuple[int, int]]:
        return (startPoint, endPoint) if startPoint[1] <= endPoint[1] else (endPoint, startPoint)

    if leftFinal is not None:
        _, (startPoint, endPoint) = leftFinal
        payload["left_line"] = {
            "start": [int(startPoint[0]), int(startPoint[1])],
            "end": [int(endPoint[0]), int(endPoint[1])],
        }

    if rightFinal is not None:
        _, (startPoint, endPoint) = rightFinal
        payload["right_line"] = {
            "start": [int(startPoint[0]), int(startPoint[1])],
            "end": [int(endPoint[0]), int(endPoint[1])],
        }

    if leftFinal is not None and rightFinal is not None:
        _, (leftStart, leftEnd) = leftFinal
        _, (rightStart, rightEnd) = rightFinal
        leftTop, leftBottom = orderedByY(leftStart, leftEnd)
        rightTop, rightBottom = orderedByY(rightStart, rightEnd)

        middleStart = [
            int(round((leftTop[0] + rightTop[0]) / 2)),
            int(round((leftTop[1] + rightTop[1]) / 2)),
        ]
        middleEnd = [
            int(round((leftBottom[0] + rightBottom[0]) / 2)),
            int(round((leftBottom[1] + rightBottom[1]) / 2)),
        ]

        payload["middle_line"] = {
            "start": middleStart,
            "end": middleEnd,
        }

    os.makedirs(os.path.dirname(outputPath), exist_ok=True)
    with open(outputPath, "w", encoding="utf-8") as jsonFile:
        json.dump(payload, jsonFile, indent=2)


def resolveVideoPath(videoName: str) -> str:
    if os.path.isabs(videoName):
        return videoName
    return os.path.join(DATA_DIR, "normal_traffic", videoName)


def runLaneDetection(
    videoName: str = DEFAULT_VIDEO_NAME,
    showVideo: bool = True,
    PASSES_COUNT: int = SAVE_LINES_AFTER_PASSES,
) -> str:
    videoPath = resolveVideoPath(videoName)

    model = YOLO(YOLO_MODEL_PATH)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    cap = cv2.VideoCapture(videoPath)
    if not cap.isOpened():
        raise FileNotFoundError(f"Could not open video: {videoPath}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(START_TIME * fps))

    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*VIDEO_FOURCC)
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, int(fps), (frameWidth, frameHeight))
    
    trajectories = defaultdict(list)
    leftBottomCornerPoints = defaultdict(list)
    rightBottomCornerPoints = defaultdict(list)
    finishedTrackLinesLeft = {}
    finishedTrackLinesRight = {}
    lastSeen = {}
    frameIndex = 0
    completedPasses = 0
    linesSaved = False

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            results = model.track(
                source=frame,
                imgsz=IMGSZ, 
                conf=CONF_THRESHOLD,
                persist=True,
                verbose=False,
                device=0 if device == 'cuda' else 'cpu',
                tracker='bytetrack.yaml',
                classes=ALLOWED_CLASSES_IDS
            )

            annotatedFrame = frame.copy()

            if results[0].boxes.id is not None:
                boxesXyxy = results[0].boxes.xyxy.cpu().numpy()
                trackIds = results[0].boxes.id.int().cpu().tolist()
                confidences = results[0].boxes.conf.cpu().tolist()

                for boxXyxy, trackId, conf in zip(boxesXyxy, trackIds, confidences):
                    centerX = int((boxXyxy[0] + boxXyxy[2]) / 2)
                    centerY = int((boxXyxy[1] + boxXyxy[3]) / 2)
                    leftBottomX = int(boxXyxy[0])
                    rightBottomX = int(boxXyxy[2])
                    bottomY = int(boxXyxy[3])

                    trajectories[trackId].append((centerX, centerY))
                    leftBottomCornerPoints[trackId].append((leftBottomX, bottomY))
                    rightBottomCornerPoints[trackId].append((rightBottomX, bottomY))
                    lastSeen[trackId] = frameIndex

                    drawCustomBox(annotatedFrame, boxXyxy, trackId, conf)

                    points = np.array(trajectories[trackId], dtype=np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotatedFrame, [points], isClosed=False, color=TRACK_COLOR, thickness=LINE_THICKNESS)

            staleIds = [carId for carId, lastFrame in lastSeen.items() if lastFrame < frameIndex - MAX_MISSING_FRAMES]
            for carId in staleIds:
                completedPasses += 1

                leftPoints = leftBottomCornerPoints[carId] if APPROX_CORNER_LEFT == "left" else rightBottomCornerPoints[carId]
                leftApproximatedLine = approximateLine(leftPoints, frameWidth, frameHeight, invertVerticalBias=True)
                if leftApproximatedLine is not None:
                    finishedTrackLinesLeft[carId] = leftApproximatedLine

                rightPoints = rightBottomCornerPoints[carId] if APPROX_CORNER_RIGHT == "right" else leftBottomCornerPoints[carId]
                rightApproximatedLine = approximateLine(rightPoints, frameWidth, frameHeight, invertVerticalBias=False)
                if rightApproximatedLine is not None:
                    finishedTrackLinesRight[carId] = rightApproximatedLine

                del lastSeen[carId]
                if carId in trajectories:
                    del trajectories[carId]
                if carId in leftBottomCornerPoints:
                    del leftBottomCornerPoints[carId]
                if carId in rightBottomCornerPoints:
                    del rightBottomCornerPoints[carId]

            if completedPasses >= PASSES_COUNT:
                if not linesSaved:
                    saveFinalLines(
                        finishedTrackLinesLeft,
                        finishedTrackLinesRight,
                        frameHeight,
                        LINES_JSON_PATH,
                        PASSES_COUNT,
                    )
                    linesSaved = True
                    print(f"Finished. lines.json saved to: {LINES_JSON_PATH}")
                    return LINES_JSON_PATH
                break

            if staleIds:
                passesLeft = max(PASSES_COUNT - completedPasses, 0)
                print(f"Passes left: {passesLeft}")

            for trackId in leftBottomCornerPoints:
                drawPersistentCornerDots(annotatedFrame, leftBottomCornerPoints[trackId])
            for trackId in rightBottomCornerPoints:
                drawPersistentCornerDots(annotatedFrame, rightBottomCornerPoints[trackId])

            drawFinishedTrackLines(annotatedFrame, finishedTrackLinesLeft, finishedTrackLinesRight)

            out.write(annotatedFrame)
            if showVideo:
                cv2.imshow(WINDOW_NAME, annotatedFrame)
                if cv2.waitKey(WAIT_KEY_MS) & 0xFF == EXIT_KEY:
                    break
                
            frameIndex += 1

    except KeyboardInterrupt:
        pass
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()

    return LINES_JSON_PATH


if __name__ == "__main__":
    runLaneDetection()