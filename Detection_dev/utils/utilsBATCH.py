"""
THIS FILE IS A TEMPORARY NEW VERSION OF UTILS that cooperates with the updated mainBATCH.py file.
"""
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Final
from .car import position, velocity, stoppingDistance

TRACK_COLOR: Final[Tuple[int, int, int]] = (0, 255, 0)
BBOX_COLOR: Final[Tuple[int, int, int]] = (255, 255, 255)
LINE_THICKNESS: Final[int] = 1
FONT_SCALE: Final[float] = 0.4
FONT_THICKNESS: Final[int] = 1

def drawCustomBox(
    annotatedFrame: np.ndarray, 
    boxXyxy: np.ndarray, 
    trackId: int, 
    conf: float, 
    carType: str, 
    x: float,
    y: float,
    w: float,
    h: float,
    speed: float,
    distance: float,
) -> None:
    x1, y1, x2, y2 = map(int, boxXyxy)
    
    line1 = f"{carType.upper()} ID:{trackId} | {conf:.2f}"
    line2 = f"x: {x:.1f}m y: {y:.1f}m | S: {speed:.1f}m/s"
    line3 = f"w: {w:.2f}m h: {h:.2f}m | B: {distance:.2f}m"

    cv2.rectangle(annotatedFrame, (x1, y1), (x2, y2), BBOX_COLOR, LINE_THICKNESS)

    font = cv2.FONT_HERSHEY_SIMPLEX
    lineHeight = int(20 * FONT_SCALE + 10)
    bgWidth = 200
    bgHeight = lineHeight * 3 + 5

    cv2.rectangle(annotatedFrame, (x1, y1 - bgHeight), (x1 + bgWidth, y1), BBOX_COLOR, -1)

    cv2.putText(annotatedFrame, line1, (x1 + 5, y1 - (lineHeight * 2) - 5), font, FONT_SCALE, (0, 0, 0), FONT_THICKNESS)
    cv2.putText(annotatedFrame, line2, (x1 + 5, y1 - lineHeight - 5), font, FONT_SCALE, (0, 0, 0), FONT_THICKNESS)
    cv2.putText(annotatedFrame, line3, (x1 + 5, y1 - 5),font, FONT_SCALE, (0, 0, 0), FONT_THICKNESS)

from typing import Dict, List, Any, Final, Tuple

MATCH_THRESHOLD_Y: Final[float] = 20.0

def matchClustersToCars(carsDict: Dict[int, Any], clusterCenters: List[Dict[str, Any]], frameIndex: int) -> float:
    allDistances: List[Tuple[float, int, int]] = []

    for carId, car in carsDict.items():
        if not car.pos:
            continue
        
        latestYolo = car.pos[-1]
        for clusterIdx, cluster in enumerate(clusterCenters):
            yCorr = cluster.get('y_corrected')
            if yCorr is None:
                continue

            distY = abs(latestYolo.y - yCorr)
            
            if distY <= MATCH_THRESHOLD_Y:
                allDistances.append((distY, carId, clusterIdx))

    allDistances.sort(key=lambda x: x[0])

    usedCars: set[int] = set()
    usedClusters: set[int] = set()
    lastDist: float = 0.0

    for dist, carId, clusterIdx in allDistances:
        if carId in usedCars or clusterIdx in usedClusters:
            continue

        cluster = clusterCenters[clusterIdx]
        
        valX = cluster.get('x_corrected')
        valY = cluster.get('y_corrected')
        valV = cluster.get('radial_velocity')

        if valX is None or valY is None or valV is None:
            continue

        car = carsDict[carId]
        radarX = float(valX)
        radarY = float(valY)
        radarV = float(valV)

        car.radarPos.append(position(x=radarX, y=radarY, frame=frameIndex))
        car.radarVel.append(velocity(v=radarV, frame=frameIndex))

        usedCars.add(carId)
        usedClusters.add(clusterIdx)
        lastDist = dist

    return lastDist
        
import cv2
import numpy as np
from typing import List, Dict, Tuple, Any, Callable

def getManualLaneLines(videoPath: str) -> List[Dict[str, float]]:
    points: List[Tuple[int, int]] = []
    windowName: str = "Select 4 points: 2 for Left Lane (Top, Bot), 2 for Right Lane (Top, Bot)"
    
    cap = cv2.VideoCapture(videoPath)
    success, frame = cap.read()
    cap.release()

    if not success:
        return []

    displayFrame = frame.copy()
    height = frame.shape[0]

    def mouseHandler(event: int, x: int, y: int, flags: int, param: Any) -> None:
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(displayFrame, (x, y), 5, (0, 0, 255), -1)
            cv2.imshow(windowName, displayFrame)

    cv2.namedWindow(windowName)
    cv2.setMouseCallback(windowName, mouseHandler)
    cv2.imshow(windowName, displayFrame)

    while len(points) < 4:
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()

    if len(points) < 4:
        return []

    detectedLines: List[Dict[str, float]] = []
    
    for i in range(0, 4, 2):
        p1, p2 = points[i], points[i+1]
        m = (p2[0] - p1[0]) / (p2[1] - p1[1]) if (p2[1] - p1[1]) != 0 else 0.0
        b = p1[0] - m * p1[1]
        xBot = m * height + b
        
        detectedLines.append({
            'm': float(m),
            'b': float(b),
            'x_bot': float(xBot),
            'abs_m': float(abs(m))
        })

    print(f"\ndetected_lines = {detectedLines}\n")
    return detectedLines

def plotRadarComparison(minX: float, maxX: float, minY: float, maxY: float, carsDict: Dict[int, Any], clusterCenters: List[Dict[str, Any]]) -> None:
    if not plt.fignum_exists(1):
        plt.figure(1, figsize=(8, 8))
    else:
        plt.figure(1)
        
    plt.clf()
    ax = plt.gca()

    radarXData = [center['x_corrected'] for center in clusterCenters]
    radarYData = [center['y_corrected'] for center in clusterCenters]
    
    carXData = [car.pos[-1].x for car in carsDict.values() if car.pos]
    carYData = [car.pos[-1].y for car in carsDict.values() if car.pos]

    plt.scatter(radarXData, radarYData, c='blue', label='Radar Centers', s=100, alpha=0.7)
    plt.scatter(carXData, carYData, c='red', marker='x', label='YOLO Cars', s=100)

    ax.set_xlim(minX, maxX)
    ax.set_ylim(minY, maxY)
    ax.set_aspect('equal', adjustable='box')

    plt.xlabel('X [m]')
    plt.ylabel('Y [m]')
    plt.legend(loc='upper right')
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.draw()
    plt.pause(0.01)


import csv
import os

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Final

PLOT_WINDOW_SIZE: Final[tuple[int, int]] = (14, 9)
TREND_LINE_DEGREE: Final[int] = 2
FIT_RESOLUTION: Final[int] = 300
POINT_ALPHA: Final[float] = 0.5

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Final, Tuple

Z_THRESHOLD: Final[float] = 2.0
PLOT_SIZE: Final[Tuple[int, int]] = (12, 8)
RESOLUTION: Final[int] = 300

def plotYOffsetCorrelation(csvPath: str) -> Callable[[float], float]:
    dataFrame = pd.read_csv(csvPath)
    
    validData = dataFrame[dataFrame['radar_frame'] == dataFrame['frame']].dropna(subset=['radar_pos_y', 'pos_diff_y'])
    
    if validData.empty:
        return lambda x: 0.0

    yValues = validData['pos_diff_y']
    cleanData = validData[np.abs(yValues - yValues.mean()) <= (Z_THRESHOLD * yValues.std())]
    
    if cleanData.empty:
        return lambda x: 0.0

    xPoints = cleanData['radar_pos_y'].to_numpy()
    yPoints = cleanData['pos_diff_y'].to_numpy()

    slope, intercept = np.polyfit(xPoints, yPoints, 1)
    
    xAxis = np.linspace(xPoints.min(), xPoints.max(), RESOLUTION)
    yAxis = slope * xAxis + intercept

    plt.figure(figsize=PLOT_SIZE)
    plt.scatter(xPoints, yPoints, color='royalblue', alpha=0.4, label='Filtered Data')
    plt.plot(xAxis, yAxis, color='crimson', linewidth=2, label=f'Linear Fit: y = {slope:.4f}x + {intercept:.4f}')
    
    plt.xlabel('Radar Position Y [m]')
    plt.ylabel('Y Offset [m]')
    plt.title('Filtered Linear Approximation')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.show()

    return lambda x: float(slope * x + intercept)


def save_car_to_csv(car, trackId, frameIndex, csv_file="cars_data.csv"):


    if frameIndex == 0:
        with open(csv_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "frame",
                "track_id",
                "pos_x", "pos_y",
                "velo",
                "radar_pos_x",
                "radar_pos_y",
                "radar_frame",
                "radar_velo",
                "pos_diff_x", "pos_diff_y",
                "velo_diff",
            ])

    if car is None:
        return

    pos = car.pos[-1] 
    velo = car.velo[-1]
    radar_pos = car.radarPos[-1] 
    radar_vel = car.radarVel[-1] 


    with open(csv_file, "a", newline="") as f:
        writer = csv.writer(f)

        writer.writerow([
            frameIndex,
            trackId,
            pos.x, 
            pos.y, 
            velo.v, 
            radar_pos.x,
            radar_pos.y, 
            radar_pos.frame,
            radar_vel.v, 
            car.posDifference.x,
            car.posDifference.y,
            car.veloDifference.v,
        ])



# TODO handle it via arg or smth , TO JEST FUNKCJ DO WYŚWITLANIA LINI 
# Draw lines for testing

# LANE_LINE_COLOR = (0, 200, 255)
# LANE_LINE_THICKNESS = 2
# y_top = 0
# y_bottom = frameHeight - 1
# for line in detected_lines:
#     m = line.get('m')
#     b = line.get('b')
#     if m is None or b is None:
#         continue

#     x_top = int((m * y_top) + b)
#     x_bottom = int((m * y_bottom) + b)
#     cv2.line(
#         annotatedFrame,
#         (x_top, y_top),
#         (x_bottom, y_bottom),
#         LANE_LINE_COLOR,
#         LANE_LINE_THICKNESS,
#     )