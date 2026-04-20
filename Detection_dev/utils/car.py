from dataclasses import dataclass
import cv2
import numpy as np
from .radar import Radar
from typing import Dict, List, Tuple, Any, Final, Union

IMAGE_WIDTH: Final[int] = 128
IMAGE_HEIGHT: Final[int] = 128
IMG_SIZE: Final[Tuple[int, int]] = (IMAGE_WIDTH, IMAGE_HEIGHT)
NORM_FACTOR: Final[float] = 255.0
posGlobalYDifference = 28.0


##TODO zmeinić na 3 kategorie najprawbopodobniej
CATEGORY_MAP: Final[Dict[int, str]] = {
    0: "coupe",
    1: "hatchback",
    2: "sedan",
    3: "suv",
    4: "truck",
    5: "van",
}

@dataclass
class position:
    x: float
    y: float
    frame: int

@dataclass
class velocity:
    v: float
    frame: int

class Car:
    def __init__(self, trackId: int):
        self.trackId = trackId
        self.x = 0.0
        self.y = 0.0
        self.w = 0.0
        self.h = 0.0

        self.radarPos: List[position] = []
        self.radarVel: List[velocity] = []
        self.pos: List[position] = []

        self.velo: List[velocity] = []
        
        self.posGlobalYDifference = 0.0
        self.posDifference: position = position(x=0.0, y=0.0, frame=-1)
        self.veloDifference: velocity = velocity(v=0.0, frame=-1)

        self.radarPos.append(position(x=0.0, y=0.0, frame=-1))
        self.radarVel.append(velocity(v=0.0, frame=-1))
        self.pos.append(position(x=0.0, y=0.0, frame=-1))
        self.velo.append(velocity(v=0.0, frame=-1))

        self.history: List[Tuple[float, float]] = []

        self.maxConfidence = 0.0
        self.lastConfidence = 0.0
        self.lastCrop = None
        self.lastSeen = -1
        self.updateCount = 0
        self.type = "unknown"
        self.mass = 0.0 ##TODO dodać methode która zminia mase ze względy na
        self.deaccelaretion = 0.0 ##TODO dodać methode która daje deacceleracje samochodu 
        self.breakingDistance = 0.0
        self.fov = 0.0
 
        self.frame_height = 0.0
        self.frame_index = 0
        self.frame_width = 0.0
        self.imgSize = 0.0
        self.radar = None

        self.wasInsideLane: Union[bool, None] = None
        self.isOutsideLane = False
        self.laneDepartureFrame = -1
        self.laneDepartureCount = 0

    def _laneBoundsAtY(self, detectedLines: List[Any], yValue: float) -> Union[Tuple[float, float], None]:
        if len(detectedLines) < 2:
            return None

        leftLine = detectedLines[0]
        rightLine = detectedLines[1]

        leftM = leftLine.get('m')
        leftB = leftLine.get('b')
        rightM = rightLine.get('m')
        rightB = rightLine.get('b')

        if None in (leftM, leftB, rightM, rightB):
            return None

        xLeft = (leftM * yValue) + leftB
        xRight = (rightM * yValue) + rightB

        laneLeft = float(min(xLeft, xRight))
        laneRight = float(max(xLeft, xRight))

        if abs(laneRight - laneLeft) < 1e-6:
            return None

        return laneLeft, laneRight

    def updateLaneState(self, detectedLines: List[Any], frameIndex: int) -> bool:
        laneBounds = self._laneBoundsAtY(detectedLines, self.y)
        if laneBounds is None:
            return False

        centerX = float(self.x)
        isInsideLane = laneBounds[0] <= centerX <= laneBounds[1]
        laneDepartureDetected = (self.wasInsideLane is True) and (not isInsideLane)

        if laneDepartureDetected:
            self.laneDepartureFrame = frameIndex
            self.laneDepartureCount += 1

        self.wasInsideLane = isInsideLane
        self.isOutsideLane = not isInsideLane

        return laneDepartureDetected

    def calcDistance(self, detectedLines: List[Any], roadWidthH0Px: float, roadWidthMeters: float) -> float:
        yBottom = self.y + (self.h / 2.0)
        xLeft = (detectedLines[0]['m'] * yBottom) + detectedLines[0]['b']
        xRight = (detectedLines[1]['m'] * yBottom) + detectedLines[1]['b']
        pixelWidthAtY = abs(xRight - xLeft)
        
        if pixelWidthAtY == 0:
            return 0.0
        
        distance = (roadWidthMeters * roadWidthH0Px) * self.fov / pixelWidthAtY
        return float(distance)
    
    def calcPosition(self, detectedLines: List[Any], roadWidthH0Px: float) -> Tuple[float, float]:
        yBottom = self.y + (self.h / 2.0)
        xCar = self.x
        
        xLeft = (detectedLines[0]['m'] * yBottom) + detectedLines[0]['b']
        xRight = (detectedLines[1]['m'] * yBottom) + detectedLines[1]['b']
        laneWidthPx = xRight - xLeft
        
        relativePos = (xCar - xLeft) / laneWidthPx if abs(laneWidthPx) > 1e-6 else 0.5
        laneWidthMeters = abs(self.radar.maxX - self.radar.minX)

        x = self.radar.minX + (relativePos * laneWidthMeters)
        y = self.calcDistance(detectedLines, roadWidthH0Px, laneWidthMeters)

        return x, y

    def checkType(self, frame: np.ndarray, cnnModel: Any) -> Tuple[str, float, np.ndarray, float]:
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, IMG_SIZE)
        imgArr = np.array(img, dtype=np.float32) / NORM_FACTOR
        imgBatch = np.expand_dims(imgArr, axis=0)

        classProbs, kPred = cnnModel.predict(imgBatch, verbose=0)
        classProbs = classProbs[0]
        kValue = float(kPred[0][0])
        predIdx = int(np.argmax(classProbs))
        confidence = float(classProbs[predIdx])
        category = CATEGORY_MAP[predIdx]

        return category, confidence, classProbs, kValue

    def update(self, box: Tuple[float, float, float, float], confidence: float, frame: np.ndarray, frameIndex: int, cnnModel: Any, 
                detectedLines: Any, roadWidthH0Px: float, fov: float, frameTime: float, imgSize: int, radar: Radar) -> None:
        
        self.x, self.y, self.w, self.h = box
        self.history.append((float(self.x), float(self.y)))
        self.frame_index = frameIndex
        self.frame_height = frame.shape[0]
        self.frame_width = frame.shape[1]
        self.lastConfidence = confidence
        self.lastSeen = frameIndex
        self.imgSize = imgSize 
        self.fov = fov
        self.radar = radar

        rawX, rawY = self.calcPosition(detectedLines, roadWidthH0Px)
        currentV = self.velo[-1].v

        if self.radarPos[-1].frame == frameIndex:
            latestRadarPos = self.radarPos[-1]
            latestRadarVel = self.radarVel[-1]

            diffX = latestRadarPos.x - rawX
            diffY = latestRadarPos.y - (rawY + posGlobalYDifference)
            self.posDifference = position(x=diffX, y=diffY, frame=frameIndex)

            currentV = latestRadarVel.v
            self.veloDifference = velocity(v=0.0, frame=frameIndex)

        finalPos = position(
            x=rawX + self.posDifference.x,
            y=rawY + posGlobalYDifference + self.posDifference.y,
            frame=frameIndex
        )
        
        self.pos.append(finalPos)
        self.velo.append(velocity(v=currentV, frame=frameIndex))

        # self.breakingDistance()

        self.updateCount += 1
        
        if confidence > self.maxConfidence:
            self.maxConfidence = confidence
            x1, y1 = int(self.x - self.w / 2), int(self.y - self.h / 2)
            x2, y2 = int(self.x + self.w / 2), int(self.y + self.h / 2)

            crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
            if crop.size > 0:
                self.lastCrop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                category, _, _, k = self.checkType(crop, cnnModel)
                self.type = category
                self.k = k

        self.breakingDistance = self.calcBreakingDistance()

    def calcBreakingDistance(self) -> float:
        ##TODO dodać breaking disancew bazując na self.pos i self.velo
        self.breakingDistance = 0.0
        return 0.0