import os
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt

RELATIVE_CSV_PATH = "..\\..\\data\\Radary_Batch\\TRUGRD_LR_like\\scenario_lane_change\\radar_points_world.csv"
COLUMN_X = "x_sensor"
COLUMN_Y = "y_sensor"
COLUMN_Z = "z_sensor"
COLUMN_VELOCITY = "radial_velocity"

SENSOR_PITCH_DEG = 0.0
SENSOR_YAW_DEG = 0.0
SENSOR_ROLL_DEG = 0.0
CAMERA_HEIGHT_OFFSET = 6.0

MASK_Z_MIN = 30.0
MASK_Z_MAX = 50
MASK_Y_MIN = 0.0
MASK_Y_MAX = 120.0

CORNER_OFFSET = 0.3
NOISE_MAX_DISTANCE = 10

POINT_SIZE = 2
OPACITY_LEVEL = 0.5
PLOT_TITLE = "Road and Moving Radar Points Visualization"
RENDERER_TYPE = "browser"

X_COLUMN = "x_corrected"
Y_COLUMN = "y_corrected"
Z_COLUMN = "z_corrected"

INTERPOLATION_FACTOR = 2

Z_AXIS_LIMIT_MIN = 0
Z_AXIS_LIMIT_MAX = 5

HISTOGRAM_BINS = 50
HIST_COLOR = "salmon"
HIST_EDGE_COLOR = "black"
HIST_TITLE = "Histogram of Points along X-axis"
HIST_LABEL_X = "Width (X)"
HIST_LABEL_COUNT = "Number of Points"

CORNER_PLOT_TITLE = "2D Projection of Road Lines"
LINE_COLOR = "red"
CENTERLINE_COLOR = "yellow"
MARKER_COLOR = "blue"
LINE_WIDTH_VIS = 3

class Street:
    def __init__(self, relativePath: str) -> None:
        self.relativePath = relativePath
        self.loadData()

    def loadData(self) -> None:
        basePath = os.path.dirname(os.path.abspath(__file__))
        self.csvPath = os.path.abspath(os.path.join(basePath, self.relativePath))
        if not os.path.exists(self.csvPath):
            raise FileNotFoundError(f"File not found: {self.csvPath}")
        self.dataFrame = pd.read_csv(self.csvPath)

    def addNoise(self, maxStep: float) -> None:
        if self.dataFrame.empty:
            return

        x = self.dataFrame[COLUMN_X].values
        y = self.dataFrame[COLUMN_Y].values
        z = self.dataFrame[COLUMN_Z].values

        norms = np.sqrt(x**2 + y**2 + z**2)
        
        # Unikamy dzielenia przez zero dla punktu (0,0,0)
        norms[norms == 0] = 1.0

        dirX = x / norms
        dirY = y / norms
        dirZ = z / norms

        noiseSteps = np.random.uniform(-maxStep, maxStep, size=len(self.dataFrame))

        self.dataFrame[COLUMN_X] += dirX * noiseSteps
        self.dataFrame[COLUMN_Y] += dirY * noiseSteps
        self.dataFrame[COLUMN_Z] += dirZ * noiseSteps

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
        deltaZ = farLowest[COLUMN_Z].mean() - closeLowest[COLUMN_Z].mean()
        deltaY = farLowest[COLUMN_Y].mean() - closeLowest[COLUMN_Y].mean()
        return float(np.degrees(np.arctan2(deltaZ, deltaY)))

    def adjustPoints(self, pitch: float, yaw: float, roll: float, heightOffset: float) -> None:
        pitchRad = np.radians(-pitch)
        yawRad = np.radians(-yaw)
        rollValue = self.calculateRoll()
        rollRad = np.radians(-rollValue)
        cosP, sinP = np.cos(pitchRad), np.sin(pitchRad)
        cosY, sinY = np.cos(yawRad), np.sin(yawRad)
        cosR, sinR = np.cos(rollRad), np.sin(rollRad)
        x = self.dataFrame[COLUMN_X]
        y = self.dataFrame[COLUMN_Y]
        z = self.dataFrame[COLUMN_Z]
        x1 = x * cosY + y * sinY
        y1 = -x * sinY + y * cosY
        x2 = x1 * cosP + z * sinP
        z2 = -x1 * sinP + z * cosP
        self.dataFrame[X_COLUMN] = x2
        self.dataFrame[Y_COLUMN] = y1 * cosR - z2 * sinR
        self.dataFrame[Z_COLUMN] = (y1 * sinR + z2 * cosR) + heightOffset

    def applyMask(self, zMin: float, zMax: float, yMin: float, yMax: float) -> None:
        self.dataFrame = self.dataFrame[
            (self.dataFrame[Y_COLUMN] >= yMin) &
            (self.dataFrame[Y_COLUMN] <= yMax) &
            (self.dataFrame[COLUMN_VELOCITY] != 0)
        ].copy()

    def interpolateCloud(self, factor: int) -> None:
        if self.dataFrame.empty or factor <= 1:
            return
        x = self.dataFrame[X_COLUMN].values
        y = self.dataFrame[Y_COLUMN].values
        z = self.dataFrame[Z_COLUMN].values
        interpX, interpY, interpZ = [], [], []
        for i in range(len(x) - 1):
            interpX.extend(np.linspace(x[i], x[i + 1], factor))
            interpY.extend(np.linspace(y[i], y[i + 1], factor))
            interpZ.extend(np.linspace(z[i], z[i + 1], factor))
        self.dataFrame = pd.DataFrame({
            X_COLUMN: interpX,
            Y_COLUMN: interpY,
            Z_COLUMN: interpZ
        })

    def plotCornerProjection(self) -> None:
        if self.dataFrame.empty:
            return

        minX = self.dataFrame[X_COLUMN].min() - CORNER_OFFSET
        maxX = self.dataFrame[X_COLUMN].max() + CORNER_OFFSET
        minY = self.dataFrame[Y_COLUMN].min()
        maxY = self.dataFrame[Y_COLUMN].max()
        midX = (minX + maxX) / 2

        plt.figure(figsize=(10, 8))
        plt.plot([minX, minX], [minY, maxY], color=LINE_COLOR, label="Left Boundary")
        plt.plot([maxX, maxX], [minY, maxY], color=LINE_COLOR, label="Right Boundary")
        plt.plot([midX, midX], [minY, maxY], color=CENTERLINE_COLOR, linestyle="--", label="Centerline")
        
        plt.scatter(self.dataFrame[X_COLUMN], self.dataFrame[Y_COLUMN], alpha=0.1, s=1)
        plt.title(CORNER_PLOT_TITLE)
        plt.xlabel(HIST_LABEL_X)
        plt.ylabel("Distance (Y)")
        plt.legend()
        plt.grid(True)
        plt.show()

    def visualize(self) -> None:
        if self.dataFrame.empty:
            return

        minX = self.dataFrame[X_COLUMN].min() - CORNER_OFFSET
        maxX = self.dataFrame[X_COLUMN].max() + CORNER_OFFSET
        minY = self.dataFrame[Y_COLUMN].min()
        maxY = self.dataFrame[Y_COLUMN].max()
        midX = (minX + maxX) / 2
        avgZ = self.dataFrame[Z_COLUMN].mean()

        fig = px.scatter_3d(
            self.dataFrame,
            x=X_COLUMN, y=Y_COLUMN, z=Z_COLUMN,
            color=Z_COLUMN, color_continuous_scale="Turbo",
            title=PLOT_TITLE, opacity=OPACITY_LEVEL,
            range_z=[Z_AXIS_LIMIT_MIN, Z_AXIS_LIMIT_MAX]
        )

        lines = [
            ([minX, minX], [minY, maxY], "Left Boundary", "solid", LINE_COLOR),
            ([maxX, maxX], [minY, maxY], "Right Boundary", "solid", LINE_COLOR),
            ([midX, midX], [minY, maxY], "Centerline", "dash", CENTERLINE_COLOR)
        ]

        for x_pts, y_pts, name, dash, color in lines:
            fig.add_trace(go.Scatter3d(
                x=x_pts, y=y_pts, z=[avgZ, avgZ],
                mode='lines',
                line=dict(color=color, width=LINE_WIDTH_VIS, dash=dash),
                name=name
            ))

        fig.update_traces(marker=dict(size=POINT_SIZE))
        fig.update_layout(
            scene=dict(aspectmode="data", zaxis=dict(range=[Z_AXIS_LIMIT_MIN, Z_AXIS_LIMIT_MAX])),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        fig.show(renderer=RENDERER_TYPE)

if __name__ == "__main__":
    streetInstance = Street(RELATIVE_CSV_PATH)
    streetInstance.addNoise(NOISE_MAX_DISTANCE)
    streetInstance.adjustPoints(SENSOR_PITCH_DEG, SENSOR_YAW_DEG, SENSOR_ROLL_DEG, CAMERA_HEIGHT_OFFSET)
    streetInstance.applyMask(MASK_Z_MIN, MASK_Z_MAX, MASK_Y_MIN, MASK_Y_MAX)
    streetInstance.interpolateCloud(INTERPOLATION_FACTOR)
    streetInstance.plotCornerProjection()
    streetInstance.visualize()