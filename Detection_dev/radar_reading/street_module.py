import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

COLUMN_X_SENSOR = "x_sensor"
COLUMN_Y_SENSOR = "y_sensor"
COLUMN_Z_SENSOR = "z_sensor"
COLUMN_TIMESTAMP = "timestamp"
COLUMN_VELOCITY = "radial_velocity"

PITCH_DEFAULT = 0.0
YAW_DEFAULT = 0.0
ROLL_DEFAULT = 0.0
HEIGHT_OFFSET_DEFAULT = 6.0

MASK_Z_MIN = 0.15
MASK_Z_MAX = 5.0
MASK_Y_MIN = 80.0
MASK_Y_MAX = 120.0

INTERPOLATION_SAMPLES = 2
POINT_SIZE_VIS = 2
LINE_WIDTH_VIS = 4
OPACITY_VAL = 0.5
PLOT_TITLE_TEXT = "Road Boundary and Centerline Visualization"
RENDERER_NAME = "browser"
COLOR_BOUNDARY = "red"
COLOR_CENTERLINE = "yellow"
STYLE_CENTERLINE = "dash"

X_CORRECTED = "x_corrected"
Y_CORRECTED = "y_corrected"
Z_CORRECTED = "z_corrected"
IS_ROAD_FLAG = "is_road"

ANIMATION_INTERVAL_MS = 100
ANIMATION_REPEAT = True
SCATTER_SIZE = 30
MOVING_POINT_COLOR = "cyan"
BOUNDARY_COLOR_PLT = "red"
CENTERLINE_COLOR_PLT = "yellow"
CENTERLINE_STYLE_PLT = "--"
BOUNDARY_STYLE_PLT = "-"
LABEL_X_TEXT = "X Corrected"
LABEL_Y_TEXT = "Y Corrected"
LABEL_Z_TEXT = "Z Corrected"
PLOT_STYLE = "dark_background"

RELATIVE_CSV_PATH = "../../data\\batch2\\batch2\\scenario1_speeding_2cars\\run_003\\radar_points_world.csv"

FOCAL_LENGTH_VAL = 0.15
LINE_WIDTH_VAL = 2
ALPHA_VAL = 0.7
SCATTER_ALPHA = 1.0
SCATTER_EDGE_WIDTH = 0.5

ROAD_Z_STEP = 0.001
ROAD_TOLERANCE = 0.001
ROLL_WINDOW_SIZE = 20.0
ROLL_MIN_POINTS = 10

ANIM_TITLE_PREFIX = "Camera Perspective View - Time: "
Z_AXIS_MIN = 0.0
Z_AXIS_MAX = 5.0

@dataclass
class RoadLine:
    startPoint: np.ndarray
    endPoint: np.ndarray
    slopeA: float
    interceptB: float
    isCenterline: bool

class Street:
    def __init__(self, relativePath: str) -> None:
        self.relativePath = relativePath
        self.contours: list[RoadLine] = []
        self.loadData()

    def loadData(self) -> None:
        basePath = os.path.dirname(os.path.abspath(__file__))
        self.csvPath = os.path.abspath(os.path.join(basePath, self.relativePath))
        if not os.path.exists(self.csvPath):
            raise FileNotFoundError(f"File not found: {self.csvPath}")
        self.dataFrame = pd.read_csv(self.csvPath)

    def calculateRoll(self) -> float:
        if self.dataFrame.empty:
            return ROLL_DEFAULT
        yData = self.dataFrame[COLUMN_Y_SENSOR]
        yMin, yMax = yData.min(), yData.max()
        closeMask = (yData >= yMin) & (yData <= yMin + ROLL_WINDOW_SIZE)
        farMask = (yData >= yMax - ROLL_WINDOW_SIZE) & (yData <= yMax)
        closePoints = self.dataFrame[closeMask]
        farPoints = self.dataFrame[farMask]
        if closePoints.empty or farPoints.empty:
            return ROLL_DEFAULT
        closeLowest = closePoints.nsmallest(ROLL_MIN_POINTS, COLUMN_Z_SENSOR)
        farLowest = farPoints.nsmallest(ROLL_MIN_POINTS, COLUMN_Z_SENSOR)
        deltaZ = farLowest[COLUMN_Z_SENSOR].mean() - closeLowest[COLUMN_Z_SENSOR].mean()
        deltaY = farLowest[COLUMN_Y_SENSOR].mean() - closeLowest[COLUMN_Y_SENSOR].mean()
        return float(np.degrees(np.arctan2(deltaZ, deltaY)))

    def adjustPoints(self, pitch: float, yaw: float, roll: float, heightOffset: float) -> None:
        pitchRad = np.radians(-pitch)
        yawRad = np.radians(-yaw)
        rollRad = np.radians(-self.calculateRoll())
        cosP, sinP = np.cos(pitchRad), np.sin(pitchRad)
        cosY, sinY = np.cos(yawRad), np.sin(yawRad)
        cosR, sinR = np.cos(rollRad), np.sin(rollRad)
        x, y, z = self.dataFrame[COLUMN_X_SENSOR], self.dataFrame[COLUMN_Y_SENSOR], self.dataFrame[COLUMN_Z_SENSOR]
        x1 = x * cosY + y * sinY
        y1 = -x * sinY + y * cosY
        x2 = x1 * cosP + z * sinP
        z2 = -x1 * sinP + z * cosP
        self.dataFrame[X_CORRECTED] = x2
        self.dataFrame[Y_CORRECTED] = y1 * cosR - z2 * sinR
        self.dataFrame[Z_CORRECTED] = (y1 * sinR + z2 * cosR) + heightOffset

    def applyMask(self, zMin: float | None = None, zMax: float | None = None, yMin: float | None = None, yMax: float | None = None, onlyStationary: bool = False) -> None:
        mask = (self.dataFrame[COLUMN_VELOCITY] == 0) if onlyStationary else (self.dataFrame[COLUMN_VELOCITY] != 0)
        if zMin is not None: mask &= (self.dataFrame[Z_CORRECTED] >= zMin)
        if zMax is not None: mask &= (self.dataFrame[Z_CORRECTED] <= zMax)
        if yMin is not None: mask &= (self.dataFrame[Y_CORRECTED] >= yMin)
        if yMax is not None: mask &= (self.dataFrame[Y_CORRECTED] <= yMax)
        self.dataFrame = self.dataFrame[mask].copy()

    def interpolateCloud(self, factor: int) -> None:
        if self.dataFrame.empty or factor <= 1:
            return
        x, y, z = self.dataFrame[X_CORRECTED].values, self.dataFrame[Y_CORRECTED].values, self.dataFrame[Z_CORRECTED].values
        interpX, interpY, interpZ = [], [], []
        for i in range(len(x) - 1):
            interpX.extend(np.linspace(x[i], x[i + 1], factor))
            interpY.extend(np.linspace(y[i], y[i + 1], factor))
            interpZ.extend(np.linspace(z[i], z[i + 1], factor))
        self.dataFrame = pd.DataFrame({X_CORRECTED: interpX, Y_CORRECTED: interpY, Z_CORRECTED: interpZ})

    def findRoad(self) -> None:
        if self.dataFrame.empty: return
        zVals = self.dataFrame[Z_CORRECTED].values
        bins = np.arange(np.min(zVals), np.max(zVals) + ROAD_Z_STEP, ROAD_Z_STEP)
        counts, edges = np.histogram(zVals, bins=bins)
        roadHeight = (edges[np.argmax(counts)] + edges[np.argmax(counts) + 1]) / 2
        self.dataFrame[IS_ROAD_FLAG] = (zVals >= roadHeight - ROAD_TOLERANCE) & (zVals <= roadHeight + ROAD_TOLERANCE)

    def findContours(self) -> None:
        roadPoints = self.dataFrame[self.dataFrame[IS_ROAD_FLAG]]
        if roadPoints.empty: return
        minX, maxX = float(roadPoints[X_CORRECTED].min()), float(roadPoints[X_CORRECTED].max())
        minY, maxY = float(roadPoints[Y_CORRECTED].min()), float(roadPoints[Y_CORRECTED].max())
        roadZ = float(roadPoints[Z_CORRECTED].mean())
        midX = (minX + maxX) / 2
        configs = [(minX, False), (maxX, False), (midX, True)]
        self.contours = []
        for xPos, isCenter in configs:
            p1, p2 = np.array([xPos, minY, roadZ]), np.array([xPos, maxY, roadZ])
            slopeA = float('inf') if (p2[0] - p1[0]) == 0 else (p2[1] - p1[1]) / (p2[0] - p1[0])
            interceptB = xPos if slopeA == float('inf') else p1[1] - slopeA * p1[0]
            self.contours.append(RoadLine(p1, p2, slopeA, interceptB, isCenter))

    def animateSpeed(self) -> None:
        if self.dataFrame.empty:
            return

        plt.style.use(PLOT_STYLE)
        fig = plt.figure(figsize=(12, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_proj_type('persp', focal_length=FOCAL_LENGTH_VAL)
        
        uniqueTimestamps = sorted(self.dataFrame[COLUMN_TIMESTAMP].unique())
        
        allX = self.dataFrame[X_CORRECTED]
        allY = self.dataFrame[Y_CORRECTED]

        def update(frameIdx: int) -> None:
            ax.clear()
            ax.set_xlim(allX.min(), allX.max())
            ax.set_ylim(allY.min(), allY.max())
            ax.set_zlim(Z_AXIS_MIN, Z_AXIS_MAX)
            
            ax.set_xlabel(LABEL_X_TEXT)
            ax.set_ylabel(LABEL_Y_TEXT)
            ax.set_zlabel(LABEL_Z_TEXT)

            for roadLine in self.contours:
                color = CENTERLINE_COLOR_PLT if roadLine.isCenterline else BOUNDARY_COLOR_PLT
                style = CENTERLINE_STYLE_PLT if roadLine.isCenterline else BOUNDARY_STYLE_PLT
                ax.plot(
                    [roadLine.startPoint[0], roadLine.endPoint[0]],
                    [roadLine.startPoint[1], roadLine.endPoint[1]],
                    [roadLine.startPoint[2], roadLine.endPoint[2]],
                    color=color, linestyle=style, linewidth=LINE_WIDTH_VAL, alpha=ALPHA_VAL
                )

            currentTime = uniqueTimestamps[frameIdx]
            frameData = self.dataFrame[self.dataFrame[COLUMN_TIMESTAMP] == currentTime]
            
            ax.scatter(
                frameData[X_CORRECTED],
                frameData[Y_CORRECTED],
                frameData[Z_CORRECTED],
                c=MOVING_POINT_COLOR,
                s=SCATTER_SIZE,
                edgecolors='white',
                linewidth=SCATTER_EDGE_WIDTH,
                alpha=SCATTER_ALPHA
            )
            
            ax.set_title(f"{ANIM_TITLE_PREFIX}{currentTime}s", color='white')

        ani = FuncAnimation(
            fig, 
            update, 
            frames=len(uniqueTimestamps), 
            interval=ANIMATION_INTERVAL_MS, 
            repeat=ANIMATION_REPEAT
        )
        
        plt.show()

    def visualize(self) -> None:
        fig = px.scatter_3d(
            self.dataFrame,
            x=X_CORRECTED,
            y=Y_CORRECTED,
            z=Z_CORRECTED,
            color=Z_CORRECTED,
            color_continuous_scale="Turbo",
            title=PLOT_TITLE_TEXT,
            opacity=OPACITY_VAL,
            range_z=[Z_AXIS_MIN, Z_AXIS_MAX]
        )

        fig.update_traces(marker=dict(size=POINT_SIZE_VIS))

        for roadLine in self.contours:
            linePoints = np.array([roadLine.startPoint, roadLine.endPoint])
            lineColor = COLOR_CENTERLINE if roadLine.isCenterline else COLOR_BOUNDARY
            lineDash = STYLE_CENTERLINE if roadLine.isCenterline else "solid"
            lineName = "Centerline" if roadLine.isCenterline else "Boundary"

            fig.add_trace(go.Scatter3d(
                x=linePoints[:, 0],
                y=linePoints[:, 1],
                z=linePoints[:, 2],
                mode='lines',
                line=dict(color=lineColor, width=LINE_WIDTH_VIS, dash=lineDash),
                name=lineName
            ))

        fig.update_layout(
            scene=dict(
                aspectmode="data",
                zaxis=dict(range=[Z_AXIS_MIN, Z_AXIS_MAX])
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        fig.show(renderer=RENDERER_NAME)

if __name__ == "__main__":
    street = Street(RELATIVE_CSV_PATH)
    street.adjustPoints(PITCH_DEFAULT, YAW_DEFAULT, ROLL_DEFAULT, HEIGHT_OFFSET_DEFAULT)
    street.applyMask(MASK_Z_MIN, MASK_Z_MAX, MASK_Y_MIN, MASK_Y_MAX, onlyStationary=True)
    street.interpolateCloud(INTERPOLATION_SAMPLES)
    street.findRoad()
    street.findContours()

    street.visualize()

    street.loadData()
    street.adjustPoints(PITCH_DEFAULT, YAW_DEFAULT, ROLL_DEFAULT, HEIGHT_OFFSET_DEFAULT)
    street.applyMask()
    street.animateSpeed()