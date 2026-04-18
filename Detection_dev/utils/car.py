import cv2
import numpy as np
from typing import Dict, List, Tuple, Any, Final
from .utils import get_x_from_line

IMAGE_WIDTH: Final[int] = 128
IMAGE_HEIGHT: Final[int] = 128
IMG_SIZE: Final[Tuple[int, int]] = (IMAGE_WIDTH, IMAGE_HEIGHT)
NORM_FACTOR: Final[float] = 255.0
SMOOTHING_WINDOW_SIZE: Final[int] = 8

CATEGORY_MAP: Final[Dict[int, str]] = {
    0: "coupe",
    1: "hatchback",
    2: "sedan",
    3: "suv",
    4: "truck",
    5: "van",
}

class Car:
    def __init__(self, trackId: int):
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
        self.breakingDistance = 0.0
        self.distance: List[float] = []
        self.velocity: List[float] = []
        self.speed = 0.0
        self.realWidth = 0.0
        self.realHeight = 0.0
        self.frame_height = 0.0
        self.frame_width = 0.0
        self.imgSize = 0.0

    def calcDistance(self, box: Tuple[float, float, float, float], detectedLines: Any, roadWidthH0Px: float, roadWidthMeters: float, fov: float, ) -> None:
        yBottom = box[1] + (box[3] / 2) 
        relativeYBottom = yBottom / self.frame_height
        yBottom = relativeYBottom * self.frame_height
        xLeft = (detectedLines[0]['m'] * yBottom) + detectedLines[0]['b']
        xRight = (detectedLines[1]['m'] * yBottom) + detectedLines[1]['b']
        roadWidthAtY = abs(xRight - xLeft)

        if roadWidthAtY > 0:
            
            calculatedDist = roadWidthMeters * (roadWidthH0Px *fov / roadWidthAtY)
            self.distance.append(float(calculatedDist))
            
            multiplyFactor = self.frame_width / self.imgSize
            metersPerPixel = roadWidthMeters / roadWidthAtY
            self.realWidth = box[2] * metersPerPixel *  multiplyFactor
            self.realHeight = box[3] * metersPerPixel *  multiplyFactor
        elif len(self.distance) > 0:
            self.distance.append(self.distance[-1])
        else:
            self.distance.append(0.0)

    def calcVelocity(self, frameTime: float) -> float:
        if len(self.distance) < 2 or frameTime <= 0:
            self.velocity.append(0.0)
            return 0.0
            
        instantVelocity = abs(self.distance[-1] - self.distance[-2]) / frameTime
        self.velocity.append(instantVelocity)
        
        window = self.velocity[-SMOOTHING_WINDOW_SIZE:]
        self.speed = sum(window) / len(window)
        
        return self.speed

    def checkType(self, frame: np.ndarray, cnnModel: Any) -> Tuple[str, float, np.ndarray, float]:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        imgArr = np.array(img, dtype=np.float32) / NORM_FACTOR
        imgBatch = np.expand_dims(imgArr, axis=0)

        classProbs, kPred = cnnModel.predict(imgBatch, verbose=0)

        classProbs = classProbs[0]
        kPred = float(kPred[0][0])
        predIdx = int(np.argmax(classProbs))
        confidence = float(classProbs[predIdx])
        category = CATEGORY_MAP[predIdx]

        return category, confidence, classProbs, kPred

    def update(self, box: Tuple[float, float, float, float], confidence: float, frame: np.ndarray, frameIndex: int, cnnModel: Any, 
               detectedLines: Any, roadWidthH0Px: float, roadWidthMeters: float, fov: float, frameTime: float,  imgSize: int) -> None:
        
        self.imgSize = imgSize 
        self.calcDistance(box, detectedLines, roadWidthH0Px, roadWidthMeters, fov)
        self.calcVelocity(frameTime)
        self.frame_height = frame.shape[0]
        self.frame_width = frame.shape[1]
        self.updateCount += 1
        self.x, self.y, self.w, self.h = box
        self.lastConfidence = confidence
        self.lastSeen = frameIndex
        bottomCenterX = float(self.x)
        bottomCenterY = float(self.y + (self.h / 2.0))
        self.history.append((bottomCenterX, bottomCenterY))



        if confidence > self.maxConfidence:
            self.maxConfidence = confidence

            x1 = int(self.x - self.w / 2)
            y1 = int(self.y - self.h / 2)
            x2 = int(self.x + self.w / 2)
            y2 = int(self.y + self.h / 2)

            crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
            if crop.size > 0:
                self.lastCrop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                category, _, _, k = self.checkType(crop, cnnModel)
                self.type = category
                self.k = k

        self.breakingDistance = self.calcBreakingDistance()

    def calcBreakingDistance(self) -> float:
        return (self.speed ** 2) * self.k