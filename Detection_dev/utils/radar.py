import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN

RELATIVE_CSV_PATH = "..\\..\\data\\normal_traffic\\radar_points_world.csv"
COLUMN_X = "x_sensor"
COLUMN_Y = "y_sensor"
COLUMN_Z = "z_sensor"
COLUMN_VELOCITY = "radial_velocity"
COLUMN_TIME = "timestamp"

SENSOR_PITCH_DEG = 0.0
SENSOR_YAW_DEG = 0.0
SENSOR_ROLL_DEG = 0.0
CAMERA_HEIGHT_OFFSET = 6.0

MASK_Z_MIN = 30.0
MASK_Z_MAX = 50.0
MASK_Y_MIN = 0.0
MASK_Y_MAX = 120.0

CORNER_OFFSET = 0.3

POINT_SIZE = 20
OPACITY_LEVEL = 0.8

X_COLUMN = "x_corrected"
Y_COLUMN = "y_corrected"
Z_COLUMN = "z_corrected"

Z_AXIS_LIMIT_MIN = 0
Z_AXIS_LIMIT_MAX = 10

CLUSTER_EPS = 1.0
CLUSTER_MIN_SAMPLES = 1
CLUSTER_COLUMN = "cluster"

CLUSTER_SCALE_X = 0.6
CLUSTER_SCALE_Y = 0.1
CLUSTER_SCALE_Z = 0.3
CLUSTER_SCALE_VELOCITY = 0.5

FIG_SIZE_X = 12
FIG_SIZE_Y = 10
COLOR_MAP = "tab20"

LINE_COLOR = "red"
CENTERLINE_COLOR = "yellow"
LINE_WIDTH_VIS = 2

INITIAL_TIME_VALUE = 0.0
LOOP_ITERATIONS = 50
TIME_STEP_DEFAULT = 0.5

class Radar:
    def __init__(self, relativePath: str, start_time: float) -> None:
        self.relativePath: str = relativePath
        self.pointsSwap: pd.DataFrame = pd.DataFrame()
        self.currentTime: float = INITIAL_TIME_VALUE
        self.t0: float = start_time
        self.loadData()

    def loadData(self) -> None:
        basePath: str = os.path.dirname(os.path.abspath(__file__))
        self.csvPath: str = os.path.abspath(os.path.join(basePath, self.relativePath))
        if not os.path.exists(self.csvPath):
            raise FileNotFoundError(f"File not found: {self.csvPath}")
        self.dataFrame: pd.DataFrame = pd.read_csv(self.csvPath)

    def step(self, timeStep: float) -> None:
        self.pointsSwap = pd.DataFrame()
        endTime: float = self.currentTime + timeStep
        print(f"Processing time step: {self.currentTime:.2f}s to {endTime:.2f}s")
        mask = (self.dataFrame[COLUMN_TIME] >= self.currentTime) & (self.dataFrame[COLUMN_TIME] < endTime)
        self.pointsSwap = self.dataFrame[mask].copy()
        self.currentTime += timeStep

    def clusterPoints(self) -> None:
        if self.pointsSwap.empty:
            return
        
        rawFeatures: np.ndarray = self.pointsSwap[[X_COLUMN, Y_COLUMN, Z_COLUMN, COLUMN_VELOCITY]].values
        
        scalingWeights: np.ndarray = np.array([
            CLUSTER_SCALE_X, 
            CLUSTER_SCALE_Y, 
            CLUSTER_SCALE_Z, 
            CLUSTER_SCALE_VELOCITY
        ])
        
        scaledFeatures: np.ndarray = rawFeatures * scalingWeights
        
        dbscan: DBSCAN = DBSCAN(eps=CLUSTER_EPS, min_samples=CLUSTER_MIN_SAMPLES)
        labels = dbscan.fit_predict(scaledFeatures)
        
        self.pointsSwap[CLUSTER_COLUMN] = labels
        self.pointsSwap = self.pointsSwap[self.pointsSwap[CLUSTER_COLUMN] != -1].copy()

    def visualizeClusteredStep(self) -> None:
        if self.dataFrame.empty:
            return

        fig: plt.Figure = plt.figure(figsize=(FIG_SIZE_X, FIG_SIZE_Y))
        ax = fig.add_subplot(111, projection='3d')
        
        fullMinX: float = self.dataFrame[X_COLUMN].min() - CORNER_OFFSET
        fullMaxX: float = self.dataFrame[X_COLUMN].max() + CORNER_OFFSET
        fullMinY: float = self.dataFrame[Y_COLUMN].min()
        fullMaxY: float = self.dataFrame[Y_COLUMN].max()
        midX: float = (fullMinX + fullMaxX) / 2
        groundZ: float = 0.0

        ax.plot([fullMinX, fullMinX], [fullMinY, fullMaxY], [groundZ, groundZ], color=LINE_COLOR, linewidth=LINE_WIDTH_VIS, label="Left Boundary")
        ax.plot([fullMaxX, fullMaxX], [fullMinY, fullMaxY], [groundZ, groundZ], color=LINE_COLOR, linewidth=LINE_WIDTH_VIS, label="Right Boundary")
        ax.plot([midX, midX], [fullMinY, fullMaxY], [groundZ, groundZ], color=CENTERLINE_COLOR, linestyle="--", linewidth=LINE_WIDTH_VIS, label="Centerline")

        if not self.pointsSwap.empty:
            uniqueClusters: np.ndarray = self.pointsSwap[CLUSTER_COLUMN].unique()
            colors = plt.cm.get_cmap(COLOR_MAP, max(len(uniqueClusters), 1))

            for i, clusterId in enumerate(uniqueClusters):
                clusterMask = self.pointsSwap[CLUSTER_COLUMN] == clusterId
                clusterData: pd.DataFrame = self.pointsSwap[clusterMask]
                
                ax.scatter(
                    clusterData[X_COLUMN], 
                    clusterData[Y_COLUMN], 
                    clusterData[Z_COLUMN],
                    s=POINT_SIZE,
                    alpha=OPACITY_LEVEL,
                    color=colors(i),
                    label=f"Cluster {clusterId}"
                )

        ax.set_xlim(fullMinX - 1, fullMaxX + 1)
        ax.set_ylim(fullMinY - 5, fullMaxY + 5)
        ax.set_zlim(Z_AXIS_LIMIT_MIN, Z_AXIS_LIMIT_MAX)
        
        ax.set_xlabel("X (Width)")
        ax.set_ylabel("Y (Distance)")
        ax.set_zlabel("Z (Height)")
        ax.set_title(f"3D Radar Scene (T={self.currentTime:.2f}s)")
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))
        
        plt.tight_layout()
        plt.show(block=True)

    def calculateRoll(self, closeWindow: float = 20.0, farWindow: float = 20.0, numLowest: int = 10) -> float:
        if self.dataFrame.empty:
            return 0.0
        y: pd.Series = self.dataFrame[COLUMN_Y]
        yMin: float = float(y.min())
        yMax: float = float(y.max())
        closeMask = (y >= yMin) & (y <= yMin + closeWindow)
        farMask = (y >= yMax - farWindow) & (y <= yMax)
        closePoints: pd.DataFrame = self.dataFrame[closeMask]
        farPoints: pd.DataFrame = self.dataFrame[farMask]
        if closePoints.empty or farPoints.empty:
            return 0.0
        closeLowest: pd.DataFrame = closePoints.nsmallest(numLowest, COLUMN_Z)
        farLowest: pd.DataFrame = farPoints.nsmallest(numLowest, COLUMN_Z)
        deltaZ: float = float(farLowest[COLUMN_Z].mean() - closeLowest[COLUMN_Z].mean())
        deltaY: float = float(farLowest[COLUMN_Y].mean() - closeLowest[COLUMN_Y].mean())
        return float(np.degrees(np.arctan2(deltaZ, deltaY)))

    def adjustPoints(self, pitch: float, yaw: float, roll: float, heightOffset: float) -> None:
        pitchRad: float = np.radians(-pitch)
        yawRad: float = np.radians(-yaw)
        rollRad: float = np.radians(-self.calculateRoll())
        cosP, sinP = np.cos(pitchRad), np.sin(pitchRad)
        cosY, sinY = np.cos(yawRad), np.sin(yawRad)
        cosR, sinR = np.cos(rollRad), np.sin(rollRad)
        x: pd.Series = self.dataFrame[COLUMN_X]
        y: pd.Series = self.dataFrame[COLUMN_Y]
        z: pd.Series = self.dataFrame[COLUMN_Z]
        x1: pd.Series = x * cosY + y * sinY
        y1: pd.Series = -x * sinY + y * cosY
        x2: pd.Series = x1 * cosP + z * sinP
        z2: pd.Series = -x1 * sinP + z * cosP
        self.dataFrame[X_COLUMN] = x2
        self.dataFrame[Y_COLUMN] = y1 * cosR - z2 * sinR
        self.dataFrame[Z_COLUMN] = (y1 * sinR + z2 * cosR) + heightOffset

    def applyMask(self, zMin: float, zMax: float, yMin: float, yMax: float) -> None:
        self.dataFrame = self.dataFrame[
            (self.dataFrame[Y_COLUMN] >= yMin) &
            (self.dataFrame[Y_COLUMN] <= yMax) &
            (self.dataFrame[COLUMN_VELOCITY] != 0)
        ].copy()
        
        if not self.dataFrame.empty:
            # self.t0 = float(self.dataFrame[COLUMN_TIME].min())
            self.currentTime = self.t0

# if __name__ == "__main__":
#     streetInstance: Street = Street(RELATIVE_CSV_PATH)
#     streetInstance.adjustPoints(SENSOR_PITCH_DEG, SENSOR_YAW_DEG, SENSOR_ROLL_DEG, CAMERA_HEIGHT_OFFSET)
#     streetInstance.applyMask(MASK_Z_MIN, MASK_Z_MAX, MASK_Y_MIN, MASK_Y_MAX)
    
#     for _ in range(LOOP_ITERATIONS):
#         streetInstance.step(TIME_STEP_DEFAULT)
#         streetInstance.clusterPoints()
#         streetInstance.visualizeClusteredStep()