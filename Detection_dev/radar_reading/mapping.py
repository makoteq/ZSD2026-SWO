import os
from dataclasses import dataclass

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

RELATIVE_CSV_PATH = "../../data\\batch2\\batch2\\scenario1_speeding_2cars\\run_007\\radar_points_world.csv"
COLUMN_X = "x_sensor"
COLUMN_Y = "y_sensor"
COLUMN_Z = "z_sensor"

SENSOR_PITCH_DEG = 0.0
SENSOR_YAW_DEG = 0.0
SENSOR_ROLL_DEG = 0.0
CAMERA_HEIGHT_OFFSET = 6.0

MASK_Z_MIN = 0.15
MASK_Z_MAX = 5.0
MASK_Y_MIN = 70.0
MASK_Y_MAX = 120.0

INTERPOLATION_FACTOR = 2
POINT_SIZE = 2
LINE_THICKNESS = 4
OPACITY_LEVEL = 0.5
PLOT_TITLE = "Road Boundary and Centerline Visualization"
RENDERER_TYPE = "browser"
BOUNDARY_COLOR = "red"
CENTERLINE_COLOR = "yellow"
CENTERLINE_DASH_STYLE = "dash"
X_COLUMN = "x_corrected"
Y_COLUMN = "y_corrected"
Z_COLUMN = "z_corrected"
IS_ROAD_COLUMN = "is_road"
EXTENSION_FACTOR = 10

@dataclass
class RoadLine:
    startPoint: np.ndarray
    endPoint: np.ndarray
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

    def calculateRoll(self, closeWindow: float = 20.0, farWindow: float = 20.0, numLowest: int = 10) -> float:
        if self.dataFrame.empty:
            return 0.0

        y = self.dataFrame[COLUMN_Y]
        yMin = y.min()
        yMax = y.max()

        closeMask = (y >= yMin) & (y <= yMin + closeWindow)
        farMask = (y >= yMax - farWindow) & (y <= yMax)

        closePoints = self.dataFrame[closeMask]
        farPoints = self.dataFrame[farMask]

        if closePoints.empty or farPoints.empty:
            return 0.0

        closeLowest = closePoints.nsmallest(numLowest, COLUMN_Z)
        farLowest = farPoints.nsmallest(numLowest, COLUMN_Z)

        zCloseMean = closeLowest[COLUMN_Z].mean()
        zFarMean = farLowest[COLUMN_Z].mean()
        yCloseMean = closeLowest[COLUMN_Y].mean()
        yFarMean = farLowest[COLUMN_Y].mean()

        deltaZ = zFarMean - zCloseMean
        deltaY = yFarMean - yCloseMean
        return float(np.degrees(np.arctan2(deltaZ, deltaY)))

    def adjustPoints(self, pitch: float, yaw: float, roll: float, heightOffset: float) -> None:
        pitchRad = np.radians(-pitch)
        yawRad = np.radians(-yaw)
        roll = self.calculateRoll()
        rollRad = np.radians(-roll)

        cosP, sinP = np.cos(pitchRad), np.sin(pitchRad)
        cosY, sinY = np.cos(yawRad), np.sin(yawRad)
        cosR, sinR = np.cos(rollRad), np.sin(rollRad)

        x = self.dataFrame[COLUMN_X]
        y = self.dataFrame[COLUMN_Y]
        z = self.dataFrame[COLUMN_Z]

        x1 = x * cosY + y * sinY
        y1 = -x * sinY + y * cosY
        z1 = z

        x2 = x1 * cosP + z1 * sinP
        y2 = y1
        z2 = -x1 * sinP + z1 * cosP

        self.dataFrame["x_corrected"] = x2
        self.dataFrame["y_corrected"] = y2 * cosR - z2 * sinR
        self.dataFrame["z_corrected"] = (y2 * sinR + z2 * cosR) + heightOffset

    def applyMask(self, zMin: float, zMax: float, yMin: float, yMax: float) -> None:
        self.dataFrame = self.dataFrame[
            (self.dataFrame["z_corrected"] >= zMin) &
            (self.dataFrame["z_corrected"] <= zMax) &
            (self.dataFrame["y_corrected"] >= yMin) &
            (self.dataFrame["y_corrected"] <= yMax) &
            (self.dataFrame["radial_velocity"] == 0)
        ].copy()

    def interpolateCloud(self, factor: int) -> None:
        if self.dataFrame.empty or factor <= 1:
            return

        x = self.dataFrame["x_corrected"].values
        y = self.dataFrame["y_corrected"].values
        z = self.dataFrame["z_corrected"].values

        interpX = []
        interpY = []
        interpZ = []

        for i in range(len(x) - 1):
            interpX.extend(np.linspace(x[i], x[i + 1], factor))
            interpY.extend(np.linspace(y[i], y[i + 1], factor))
            interpZ.extend(np.linspace(z[i], z[i + 1], factor))

        self.dataFrame = pd.DataFrame({
            "x_corrected": interpX,
            "y_corrected": interpY,
            "z_corrected": interpZ
        })

    def findRoad(self) -> None:
        if self.dataFrame.empty:
            return

        z = self.dataFrame["z_corrected"].values
        bins = np.arange(np.min(z), np.max(z) + 0.00656, 0.00656)
        counts, _ = np.histogram(z, bins=bins)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        road_height = bin_centers[np.argmax(counts)]
        tolerance = 0.001
        self.dataFrame[IS_ROAD_COLUMN] = (z >= road_height - tolerance) & (z <= road_height + tolerance)

    def findContours(self) -> None:
        roadPoints = self.dataFrame[self.dataFrame[IS_ROAD_COLUMN]]
        if roadPoints.empty:
            return

        minX = float(roadPoints[X_COLUMN].min())
        maxX = float(roadPoints[X_COLUMN].max())
        minY = float(roadPoints[Y_COLUMN].min())
        maxY = float(roadPoints[Y_COLUMN].max())
        roadHeight = float(roadPoints[Z_COLUMN].mean())
        centerX = (minX + maxX) / 2

        self.contours = [
            RoadLine(np.array([minX, minY, roadHeight]), np.array([minX, maxY, roadHeight]), False),
            RoadLine(np.array([maxX, minY, roadHeight]), np.array([maxX, maxY, roadHeight]), False),
            RoadLine(np.array([centerX, minY, roadHeight]), np.array([centerX, maxY, roadHeight]), True)
        ]

    def visualize(self) -> None:
        if IS_ROAD_COLUMN not in self.dataFrame.columns:
            self.dataFrame[IS_ROAD_COLUMN] = False

        fig = px.scatter_3d(
            self.dataFrame,
            x=X_COLUMN,
            y=Y_COLUMN,
            z=Z_COLUMN,
            color=Z_COLUMN,
            color_continuous_scale="Turbo",
            title=PLOT_TITLE,
            opacity=OPACITY_LEVEL
        )

        fig.update_traces(marker=dict(size=POINT_SIZE))

        for roadLine in self.contours:
            direction = roadLine.endPoint - roadLine.startPoint
            extendedPoint1 = roadLine.startPoint - direction * EXTENSION_FACTOR
            extendedPoint2 = roadLine.endPoint + direction * EXTENSION_FACTOR
            linePoints = np.array([extendedPoint1, extendedPoint2])
            lineColor = CENTERLINE_COLOR if roadLine.isCenterline else BOUNDARY_COLOR
            lineDash = CENTERLINE_DASH_STYLE if roadLine.isCenterline else "solid"
            lineName = "Centerline" if roadLine.isCenterline else "Boundary"

            fig.add_trace(go.Scatter3d(
                x=linePoints[:, 0],
                y=linePoints[:, 1],
                z=linePoints[:, 2],
                mode='lines',
                line=dict(color=lineColor, width=LINE_THICKNESS, dash=lineDash),
                name=lineName
            ))

        fig.update_layout(scene=dict(aspectmode="data"), margin=dict(l=0, r=0, b=0, t=40))
        fig.show(renderer=RENDERER_TYPE)

if __name__ == "__main__":
    streetInstance = Street(RELATIVE_CSV_PATH)
    streetInstance.adjustPoints(SENSOR_PITCH_DEG, SENSOR_YAW_DEG, SENSOR_ROLL_DEG, CAMERA_HEIGHT_OFFSET)
    streetInstance.applyMask(MASK_Z_MIN, MASK_Z_MAX, MASK_Y_MIN, MASK_Y_MAX)
    streetInstance.interpolateCloud(INTERPOLATION_FACTOR)
    streetInstance.findRoad()
    streetInstance.findContours()
    streetInstance.visualize()

  

