import os
import ultralytics
import tensorflow as tf
import torch
import cv2
import numpy as np
import nbformat
import base64
import io
from ultralytics import YOLO
from PIL import Image
from keras import models
from typing import Tuple, List, Dict, Any


CURRENT_SCRIPT_PATH = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.abspath(os.path.join(CURRENT_SCRIPT_PATH, "../..", "data"))
VIDEO_PATH = os.path.join(DATA_DIR, "normal_traffic\\rgb.mp4")
YOLO_MODEL_PATH = os.path.join(DATA_DIR, "models", "best.pt")
CNN_MODEL_PATH = os.path.join(DATA_DIR, "models", "cnn.h5")
OUTPUT_VIDEO_PATH = os.path.join(DATA_DIR, "output", "trajectory.mp4")
OUTPUT_NB_PATH = os.path.join(DATA_DIR, "output", "cars_output.ipynb")
START_TIME = 1
CONF_THRESHOLD = 0.5
IMGSZ = 800
ALLOWED_CLASSES_IDS = [0]
MAX_MISSING_FRAMES = float('inf')
VIDEO_FOURCC = 'mp4v'
TRACK_COLOR = (0, 255, 0)
BBOX_COLOR = (255, 255, 255)
LINE_THICKNESS = 1
FONT_SCALE = 0.4
FONT_THICKNESS = 1
WINDOW_NAME = "Traffic Analysis"
WAIT_KEY_MS = 1
EXIT_KEY = ord('q')


class Car:
    """
    Represents a tracked vehicle in the video stream.

    Stores the vehicle's position, bounding box dimensions, tracking history,
    classification type, and kinematic properties such as speed and braking distance.
    Classification is performed using a CNN model whenever a higher-confidence
    detection is observed.
    """

    CATEGORY_MAP: Dict[int, str] = {
        0: "coupe",
        1: "hatchback",
        2: "sedan",
        3: "suv",
        4: "truck",
        5: "van",
    }
    IMG_SIZE: Tuple[int, int] = (128, 128)

    def __init__(self, trackId: int):
        """
        Initialize a Car instance with a given tracking ID.

        Args:
            trackId: Unique identifier assigned by the tracker to this vehicle.
        """
        self.trackId = trackId
        self.x = 0.0
        self.y = 0.0
        self.w = 0.0
        self.h = 0.0
        self.maxConfidence = 0.0
        self.lastConfidence = 0.0
        self.history: List[Tuple[float, float]] = []
        self.lastCrop = None
        self.lastSeen = -1
        self.updateCount = 0
        self.type = "unknown"
        self.k = 0.0
        self.speed = 0.0
        self.breakingDistance = 0.0

    def checkType(self, frame: np.ndarray) -> Tuple[str, float, np.ndarray, float]:
        """
        Classify the vehicle type from a cropped frame region using the CNN model.

        The frame is resized to IMG_SIZE, normalized to [0, 1], and passed through
        the CNN. The model returns class probabilities and a braking coefficient k.

        Args:
            frame: BGR image array containing the cropped vehicle region.

        Returns:
            A tuple of:
                - category (str): Predicted vehicle category name.
                - confidence (float): Probability of the predicted class.
                - classProbs (np.ndarray): Full array of class probabilities.
                - kPred (float): Predicted braking coefficient k.
        """
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, self.IMG_SIZE)
        imgArr = np.array(img, dtype=np.float32) / 255.0
        imgBatch = np.expand_dims(imgArr, axis=0)

        classProbs, kPred = cnn.predict(imgBatch, verbose=0)

        classProbs = classProbs[0]
        kPred = float(kPred[0][0])
        predIdx = int(np.argmax(classProbs))
        confidence = float(classProbs[predIdx])
        category = self.CATEGORY_MAP[predIdx]

        return category, confidence, classProbs, kPred

    def update(self, box: Tuple[float, float, float, float], confidence: float, frame: np.ndarray, frameIndex: int) -> None:
        """
        Update the vehicle's state with a new detection from the current frame.

        Updates position, bounding box, confidence, and tracking history. If the
        new detection has a higher confidence than any previous one, the vehicle
        crop is re-classified using the CNN and the braking coefficient is updated.
        Braking distance is recalculated after every update.

        Args:
            box: Bounding box in (cx, cy, w, h) format (center x/y plus width/height).
            confidence: YOLO detection confidence score for this detection.
            frame: Full BGR video frame from which the crop will be extracted.
            frameIndex: Index of the current video frame (used for staleness tracking).
        """
        self.updateCount += 1
        self.x, self.y, self.w, self.h = box
        self.lastConfidence = confidence
        self.lastSeen = frameIndex
        self.history.append((float(self.x), float(self.y)))

        if confidence > self.maxConfidence:
            self.maxConfidence = confidence

            x1 = int(self.x - self.w / 2)
            y1 = int(self.y - self.h / 2)
            x2 = int(self.x + self.w / 2)
            y2 = int(self.y + self.h / 2)

            crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
            if crop.size > 0:
                self.lastCrop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                category, _, _, k = self.checkType(crop)
                self.type = category
                self.k = k

        self.breakingDistance = self.calcBreakingDistance()

    def calcBreakingDistance(self) -> float:
        """
        Calculate the estimated braking distance for this vehicle.

        Uses the kinematic formula: d = v² * k, where v is the current speed
        and k is the vehicle-specific braking coefficient predicted by the CNN.

        Returns:
            Estimated braking distance in consistent units with speed and k.
        """
        return (self.speed ** 2) * self.k


def drawCustomBox(annotatedFrame: np.ndarray, boxXyxy: np.ndarray, trackId: int, conf: float, carType: str) -> None:
    """
    Draw a bounding box with a label overlay onto the annotated frame in-place.

    Renders a white rectangle around the detected vehicle and a filled label
    background displaying the vehicle type, track ID, and detection confidence.

    Args:
        annotatedFrame: BGR image array to draw onto (modified in-place).
        boxXyxy: Bounding box coordinates as [x1, y1, x2, y2].
        trackId: Unique tracker ID of the vehicle, shown in the label.
        conf: Detection confidence score, shown in the label.
        carType: Predicted vehicle category string, shown in the label.
    """
    x1, y1, x2, y2 = map(int, boxXyxy)
    labelText = f"{carType} {trackId} | {conf:.2f}"

    cv2.rectangle(annotatedFrame, (x1, y1), (x2, y2), BBOX_COLOR, LINE_THICKNESS)

    (textWidth, textHeight), _ = cv2.getTextSize(labelText, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, FONT_THICKNESS)
    cv2.rectangle(annotatedFrame, (x1, y1 - textHeight - 5), (x1 + textWidth, y1), BBOX_COLOR, -1)
    cv2.putText(annotatedFrame, labelText, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (0, 0, 0), FONT_THICKNESS)


def main() -> None:
    """
    Run the main traffic analysis pipeline.

    Opens the input video, initializes a VideoWriter for the output, and processes
    each frame in a loop:

    1. Runs YOLO tracking (ByteTrack) to detect and track vehicles.
    2. Updates each tracked Car object with the latest bounding box and frame crop.
    3. Draws bounding boxes, labels, and trajectory polylines on the annotated frame.
    4. Removes stale or low-confidence tracks from the active dictionary.
    5. Writes the annotated frame to the output video and displays it in a window.

    The loop exits when the video ends, the user presses the exit key (q), or a
    KeyboardInterrupt is received. All resources are released in the finally block.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(START_TIME * fps))

    frameWidth = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frameHeight = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fourcc = cv2.VideoWriter_fourcc(*VIDEO_FOURCC)
    out = cv2.VideoWriter(OUTPUT_VIDEO_PATH, fourcc, int(fps), (frameWidth, frameHeight))
    
    carsDict: Dict[int, Car] = {}
    frameIndex = 0

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
                boxesXywh = results[0].boxes.xywh.cpu().numpy()
                trackIds = results[0].boxes.id.int().cpu().tolist()
                confidences = results[0].boxes.conf.cpu().tolist()

                for boxXyxy, boxXywh, trackId, conf in zip(boxesXyxy, boxesXywh, trackIds, confidences):
                    if trackId not in carsDict:
                        carsDict[trackId] = Car(trackId)

                    car = carsDict[trackId]
                    car.update(boxXywh, conf, frame, frameIndex)

                    drawCustomBox(annotatedFrame, boxXyxy, trackId, conf, car.type)

                    points = np.array(car.history).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotatedFrame, [points], isClosed=False, color=TRACK_COLOR, thickness=LINE_THICKNESS)

            staleIds = [carId for carId, carObj in carsDict.items()
                        if carObj.lastSeen < frameIndex - MAX_MISSING_FRAMES
                        or carObj.lastConfidence < CONF_THRESHOLD]
            for carId in staleIds:
                del carsDict[carId]

            out.write(annotatedFrame)
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


if __name__ == "__main__":
    model = YOLO(YOLO_MODEL_PATH)
    cnn = models.load_model(CNN_MODEL_PATH, compile=False)
    main()