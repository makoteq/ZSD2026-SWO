import numpy as np
import cv2

TRACK_COLOR = (0, 255, 0)
BBOX_COLOR = (255, 255, 255)
LINE_THICKNESS = 1
FONT_SCALE = 0.4
FONT_THICKNESS = 1

def get_x_from_line(line, y):

        if isinstance(line, dict):
            m = line.get('m', 0)
            b = line.get('b', 0)
            return m * y + b

        if hasattr(line, 'get_x'):
            return line.get_x(y)
        if isinstance(line, (list, np.ndarray)) and len(line) == 2:
            return line[0] * y + line[1]
        
        return 0    

def drawCustomBox(
    annotatedFrame: np.ndarray, 
    boxXyxy: np.ndarray, 
    trackId: int, 
    conf: float, 
    carType: str, 
    distance: float, 
    speed: float,
    realW: float,
    realH: float
) -> None:
    x1, y1, x2, y2 = map(int, boxXyxy)
    
    speedKmh = speed * 3.6
    line1 = f"{carType.upper()} ID:{trackId} | {conf:.2f}"
    line2 = f"D: {distance:.1f}m | S: {speedKmh:.1f}km/h"
    line3 = f"W: {realW:.2f}m | H: {realH:.2f}m"

    cv2.rectangle(annotatedFrame, (x1, y1), (x2, y2), BBOX_COLOR, LINE_THICKNESS)

    font = cv2.FONT_HERSHEY_SIMPLEX
    lineHeight = int(20 * FONT_SCALE + 10)
    bgWidth = 180
    bgHeight = lineHeight * 3 + 5

    cv2.rectangle(annotatedFrame, (x1, y1 - bgHeight), (x1 + bgWidth, y1), BBOX_COLOR, -1)

    cv2.putText(annotatedFrame, line1, (x1 + 5, y1 - (lineHeight * 2) - 5), font, FONT_SCALE, (0, 0, 0), FONT_THICKNESS)
    cv2.putText(annotatedFrame, line2, (x1 + 5, y1 - lineHeight - 5), font, FONT_SCALE, (0, 0, 0), FONT_THICKNESS)
    cv2.putText(annotatedFrame, line3, (x1 + 5, y1 - 5), font, FONT_SCALE, (255, 0, 0), FONT_THICKNESS)

