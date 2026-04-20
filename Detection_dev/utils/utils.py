import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict, Any, Final
from .car import position, velocity


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
    speed: float,
) -> None:
    x1, y1, x2, y2 = map(int, boxXyxy)
    
    line1 = f"{carType.upper()} ID:{trackId} | {conf:.2f}"
    line2 = f"x: {x:.1f}m y: {y:.1f}m | S: {speed:.1f}m/s"

    cv2.rectangle(annotatedFrame, (x1, y1), (x2, y2), BBOX_COLOR, LINE_THICKNESS)

    font = cv2.FONT_HERSHEY_SIMPLEX
    lineHeight = int(20 * FONT_SCALE + 10)
    bgWidth = 180
    bgHeight = lineHeight * 3 + 5

    cv2.rectangle(annotatedFrame, (x1, y1 - bgHeight), (x1 + bgWidth, y1), BBOX_COLOR, -1)

    cv2.putText(annotatedFrame, line1, (x1 + 5, y1 - (lineHeight * 2) - 5), font, FONT_SCALE, (0, 0, 0), FONT_THICKNESS)
    cv2.putText(annotatedFrame, line2, (x1 + 5, y1 - lineHeight - 5), font, FONT_SCALE, (0, 0, 0), FONT_THICKNESS)

def matchClustersToCars(carsDict: Dict[int, Any], clusterCenters: List[Dict[str, Any]], frameIndex: int) -> float:
    allDistances = []

    for carId, car in carsDict.items():
        if not car.pos:
            continue
        
        latestYolo = car.pos[-1]
        for clusterIdx, cluster in enumerate(clusterCenters):
            distY = abs(latestYolo.y - cluster['y_corrected'])
            allDistances.append((distY, carId, clusterIdx))

    allDistances.sort(key=lambda x: x[0])

    usedCars = set()
    usedClusters = set()

    for dist, carId, clusterIdx in allDistances:
        if carId in usedCars or clusterIdx in usedClusters:
            continue

        car = carsDict[carId]
        cluster = clusterCenters[clusterIdx]
        
        radarX = float(cluster['x_corrected'])
        radarY = float(cluster['y_corrected'])
        radarV = float(cluster['radial_velocity'])

        car.radarPos.append(position(x=radarX, y=radarY, frame=frameIndex))
        car.radarVel.append(velocity(v=radarV, frame=frameIndex))

        usedCars.add(carId)
        usedClusters.add(clusterIdx)

        print(f"MATCHED ID: {carId:2} | Frame: {frameIndex:5} | Y-Dist: {dist:4.2f}m | X: {radarX:6.2f}m | Y: {radarY:6.2f}m")

        return dist
        
import cv2
import numpy as np
from typing import List, Dict, Tuple, Any

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