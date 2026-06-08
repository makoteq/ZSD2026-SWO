from dataclasses import dataclass
import cv2
import numpy as np
from .radar import Radar
from typing import Dict, List, Tuple, Any, Final, Union, Callable
from .weather import getWeather
# from ..algorithms.model_training.vehicle_classification_CNN.dataset import mass_factor
from datetime import date

IMAGE_WIDTH: Final[int] = 128
IMAGE_HEIGHT: Final[int] = 128
IMG_SIZE: Final[Tuple[int, int]] = (IMAGE_WIDTH, IMAGE_HEIGHT)
NORM_FACTOR: Final[float] = 255.0
OFFSET = 28.0
SMOOTHING_WINDOW_SIZE: Final[int] = 8

ROAD_WIDTH_METERS = 7.0

@dataclass
class position:
    """
    Container including position data of the vehicle in a specific frame.

    Attributes:
        x (float): X position in meters.
        y (float): Y position in meters.
        frame (int): Frame index at which the measurement was taken.
    """
    x: float
    y: float
    frame: int


@dataclass
class velocity:
    """
    Container including velocity data of the vehicle in a specific frame.

    Attributes:
        v (float): Velocity in m/s.
        frame (int): Frame index at which the measurement was taken.
    """
    v: float
    frame: int


@dataclass
class size:
    """
    Container including physical object dimensions estimated for a specific frame.

    Attributes:
        w (float): Estimated object width in meters.
        h (float): Estimated object height in meters.
        frame (int): Frame index at which the measurement was taken.
    """
    w: float
    h: float
    frame: int


@dataclass
class stoppingDistance:
    """
    Container including estimated stopping distance data of the vehicle.
    """
    distance: float
    mass: float
    car_category: str = 'medium'


class Car:
    """
    Represents a tracked vehicle.

    Stores image-space detections, radar measurements,
    estimated world coordinates, velocity, dimensions,
    stopping distance information, and lane departure state.
    """
    def __init__(self, trackId: int):
        self.trackId = trackId
        self.x = 0.0
        self.y = 0.0
        self.w = 0.0
        self.h = 0.0

        self.stoppingDistance: List[stoppingDistance] = []
        self.stoppingDistance.append(stoppingDistance(distance=0.0, mass=0.0))

        self.radarPos: List[position] = []
        self.radarVel: List[velocity] = []
        self.pos: List[position] = []
        self.velo: List[velocity] = []
        self.size: List[size] = []

        self.size.append(size(w=0.0, h=0.0, frame=-1))
        self.posDifference: position = position(x=0.0, y=0.0, frame=-1)
        self.veloDifference: velocity = velocity(v=0.0, frame=-1)

        self.radarPos.append(position(x=0.0, y=0.0, frame=-1))
        self.radarVel.append(velocity(v=0.0, frame=-1))
        self.pos.append(position(x=0.0, y=0.0, frame=-1))
        self.velo.append(velocity(v=0.0, frame=-1))

        self.correctionFunc: Callable[[float], float] = lambda x: 0.0

        self.history: List[Tuple[float, float]] = []
        self.maxConfidence = 0.0
        self.lastConfidence = 0.0
        self.lastCrop = None
        self.lastSeen = -1
        self.updateCount = 0
        self.type = "unknown"
        self.mass = 0.0
        self.deacceleration = 0.0
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
        """
        Compute lane boundary coordinates at a specified image Y position.

        Returns:
            Tuple[float, float] | None: Left and right lane boundaries in pixels.
        """
        leftLine = next((line for line in detectedLines if line.get('name') == 'left_line'), None)
        rightLine = next((line for line in detectedLines if line.get('name') == 'right_line'), None)

        if leftLine is None or rightLine is None:
            # Backward compatibility for older line dictionaries without explicit names.
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

    def updateLaneState(
            self,
            detectedLines: List[Any],
            frameIndex: int,
            centerX: Union[float, None] = None,
            centerY: Union[float, None] = None,
    ) -> bool:
        """
        Update lane occupancy state and detect lane departures.

        Returns:
            bool: True if a new lane departure event is detected.
        """
        laneCenterX = float(self.x if centerX is None else centerX)
        laneCenterY = float(self.y if centerY is None else centerY)

        laneBounds = self._laneBoundsAtY(detectedLines, laneCenterY)
        if laneBounds is None:
            return False

        isInsideLane = laneBounds[0] <= laneCenterX <= laneBounds[1]
        wasOutsideLane = self.wasInsideLane is False
        laneDepartureDetected = (not isInsideLane) and (not wasOutsideLane)

        if laneDepartureDetected:
            self.laneDepartureFrame = frameIndex
            self.laneDepartureCount += 1

        self.wasInsideLane = isInsideLane
        self.isOutsideLane = not isInsideLane

        return laneDepartureDetected

    def getCorrectedDistance(self, cameraDist: float) -> float:
        """
        Apply distance correction to a camera-based estimate.
        """
        offset = self.correctionFunc(cameraDist)
        return float(cameraDist + offset)

    def calcDistance(self, detectedLines: List[Any], roadWidthH0Px: float, roadWidthMeters: float) -> float:
        """
        Estimate longitudinal distance using lane-width perspective scaling.

        Returns:
            float: Estimated distance in meters.
        """
        yBottom = self.y + (self.h / 2.0)
        xLeft = (detectedLines[0]['m'] * yBottom) + detectedLines[0]['b']
        xRight = (detectedLines[1]['m'] * yBottom) + detectedLines[1]['b']
        pixelWidthAtY = abs(xRight - xLeft)

        if pixelWidthAtY == 0:
            return 0.0

        D_0 = 8.5 
        distance = D_0 * (roadWidthH0Px / pixelWidthAtY)
        return float(distance)

    def getSize(self, detectedLines: List[Any], roadWidthH0Px: float) -> Tuple[float, float]:
        """
        Estimate the real-world dimensions of the tracked object.

        The calculation uses the detected lane geometry, estimated object
        distance, camera field of view, and bounding-box dimensions.

        Returns:
            Tuple[float, float]: Estimated width and height in meters.
        """
        yBottom = self.y + (self.h / 2.0)

        xLeft = (detectedLines[0]['m'] * yBottom) + detectedLines[0]['b']
        xRight = (detectedLines[1]['m'] * yBottom) + detectedLines[1]['b']

        laneWidthPx = abs(xRight - xLeft)
        if laneWidthPx < 1e-6:
            return 0.0, 0.0

        laneWidthMeters = abs(self.radar.maxX - self.radar.minX)

        distance = self.getCorrectedDistance(
            self.calcDistance(detectedLines, roadWidthH0Px, laneWidthMeters)
        )

        if distance <= 0:
            return 0.0, 0.0

        fov_rad = np.radians(self.fov if self.fov > 0 else 60.0)

        scene_width = 2.0 * distance * np.tan(fov_rad / 2.0)

        box_width_ratio = self.w / max(self.frame_width, 1)
        box_height_ratio = self.h / max(self.frame_height, 1)

        w_m = box_width_ratio * scene_width

        h_m = box_height_ratio * scene_width

        return float(w_m), float(h_m)

    def calcPosition(self, detectedLines: List[Any], roadWidthH0Px: float) -> Tuple[float, float]:
        """
        Estimate the object's position in road coordinates.

        The lateral position is computed relative to the detected lane
        boundaries, while the longitudinal position is estimated from
        perspective geometry.

        Returns:
            Tuple[float, float]: Position (x, y) in meters.
        """
        yBottom = self.y + (self.h / 2.0)
        xCar = self.x

        xLeft = (detectedLines[0]['m'] * yBottom) + detectedLines[0]['b']
        xRight = (detectedLines[1]['m'] * yBottom) + detectedLines[1]['b']
        laneWidthPx = xRight - xLeft

        relativePos = (xCar - xLeft) / laneWidthPx if abs(laneWidthPx) > 1e-6 else 0.5
        laneWidthMeters = abs(self.radar.maxX - self.radar.minX)

        x = self.radar.minX + (relativePos * laneWidthMeters)
        y = self.getCorrectedDistance(self.calcDistance(detectedLines, roadWidthH0Px, laneWidthMeters))

        return float(x), float(y)

    def update(self, box: Tuple[float, float, float, float], confidence: float, frame: np.ndarray, frameIndex: int,
               detectedLines: Any, roadWidthH0Px: float, fov: float, frameTime: float, imgSize: int, radar: Radar,
               CorrectionFunc: Callable[[float], float]) -> None:
        """
        Update object state using the latest detection.

        Updates image-space coordinates, estimated world position,
        physical dimensions, confidence score, and tracking metadata.

        Parameters:
            box: Bounding box coordinates.
            confidence: Detection confidence score.
            frame: Current video frame.
            frameIndex: Frame index.
            detectedLines: Detected lane boundaries.
            roadWidthH0Px: Road width in image coordinates.
            fov: Camera field of view.
            frameTime: Frame duration.
            imgSize: Input image size used for detection.
            radar: Radar reference data.
            CorrectionFunc: Distance correction function.
        """
        self.x, self.y, self.w, self.h = box
        self.history.append((float(self.x), float(self.y)))
        self.frame_index = frameIndex
        self.correctionFunc = CorrectionFunc
        self.frame_height = frame.shape[0]
        self.frame_width = frame.shape[1]
        self.lastConfidence = confidence
        self.lastSeen = frameIndex
        self.imgSize = imgSize
        self.fov = fov
        self.radar = radar

        rawX, rawY = self.calcPosition(detectedLines, roadWidthH0Px)
        currentV = float(self.velo[-1].v)

        realW, realH = self.getSize(detectedLines, roadWidthH0Px)
        self.size.append(
            size(
                w=float(realW),
                h=float(realH),
                frame=frameIndex
            )
        )

        if self.radarPos[-1].frame == frameIndex:
            latestRadarPos = self.radarPos[-1]
            latestRadarVel = self.radarVel[-1]

            diffX = float(latestRadarPos.x - rawX)
            diffY = float(latestRadarPos.y - (rawY + OFFSET))
            self.posDifference = position(x=diffX, y=diffY, frame=frameIndex)

            currentV = float(latestRadarVel.v)
            self.veloDifference = velocity(v=0.0, frame=frameIndex)

        finalPos = position(
            x=float(rawX + self.posDifference.x),
            y=float(rawY + self.posDifference.y + OFFSET),
            frame=frameIndex
        )

        self.pos.append(finalPos)
        self.velo.append(velocity(v=currentV, frame=frameIndex))

        self.updateCount += 1

        if confidence > self.maxConfidence:
            self.maxConfidence = confidence
            x1, y1 = int(self.x - self.w / 2), int(self.y - self.h / 2)
            x2, y2 = int(self.x + self.w / 2), int(self.y + self.h / 2)

            crop = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
            if crop.size > 0:
                self.lastCrop = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)


