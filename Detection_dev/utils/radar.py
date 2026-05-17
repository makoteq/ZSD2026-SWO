import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from typing import List, Dict

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

CORNER_OFFSET = 0.25

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

MAX_DISTANCE_ERROR = 0.5
MAX_AZIMUTH_ERROR_DEG = 0.25
MAX_ELEVATION_ERROR_DEG = 0.25
MAX_VELOCITY_ERROR = 0.1

INITIAL_TIME_VALUE = 0.0
LOOP_ITERATIONS = 50
TIME_STEP_DEFAULT = 0.5

class Radar:
    def __init__(self, relativePath: str, start_time: float) -> None:
        self.relativePath: str = relativePath
        self.pointsSwap: np.ndarray = np.empty((0, 4), dtype=float)
        self.pointsSwapLabels: np.ndarray = np.empty((0,), dtype=int)
        self.currentTime: float = INITIAL_TIME_VALUE
        self.lane_width_meters: float = 0.0
        self.t0: float = start_time
        self.minX: float = 0.0
        self.maxX: float = 0.0
        self.minY: float = 0.0
        self.maxY: float = 0.0
        self.clusterCenters: List[Dict[str, float]] = []
        self._timeValues: np.ndarray = np.empty((0,), dtype=float)
        self._pointsValues: np.ndarray = np.empty((0, 4), dtype=float)
        self.debug: bool = False
        self.loadData()
        self.adjustPoints(SENSOR_PITCH_DEG, SENSOR_YAW_DEG, SENSOR_ROLL_DEG, CAMERA_HEIGHT_OFFSET)
        self._refresh_cache()
        
   
    def loadData(self) -> None:
        basePath: str = os.path.dirname(os.path.abspath(__file__))
        self.csvPath: str = os.path.abspath(os.path.join(basePath, self.relativePath))
        if not os.path.exists(self.csvPath):
            raise FileNotFoundError(f"File not found: {self.csvPath}")
        self.dataFrame: pd.DataFrame = pd.read_csv(self.csvPath)
        self.dataFrame.sort_values(COLUMN_TIME, inplace=True)
        self.dataFrame.reset_index(drop=True, inplace=True)

    def _refresh_cache(self) -> None:
        if self.dataFrame.empty:
            self._timeValues = np.empty((0,), dtype=float)
            self._pointsValues = np.empty((0, 4), dtype=float)
            return
        self._timeValues = self.dataFrame[COLUMN_TIME].to_numpy()
        self._pointsValues = self.dataFrame[[X_COLUMN, Y_COLUMN, Z_COLUMN, COLUMN_VELOCITY]].to_numpy()

    def step(self, timeStep: float) -> None:
        self.pointsSwap = np.empty((0, 4), dtype=float)
        self.pointsSwapLabels = np.empty((0,), dtype=int)
        if self._timeValues.size == 0:
            return
        endTime: float = self.currentTime + timeStep
        if self.debug:
            print(f"Processing time step: {self.currentTime:.2f}s to {endTime:.2f}s")
        start_idx = np.searchsorted(self._timeValues, self.currentTime, side="left")
        end_idx = np.searchsorted(self._timeValues, endTime, side="left")
        if end_idx > start_idx:
            self.pointsSwap = self._pointsValues[start_idx:end_idx]
        self.currentTime = endTime

    def clusterPoints(self) -> None:
        if self.pointsSwap.size == 0:
            return

        scalingWeights: np.ndarray = np.array([
            CLUSTER_SCALE_X,
            CLUSTER_SCALE_Y,
            CLUSTER_SCALE_Z,
            CLUSTER_SCALE_VELOCITY,
        ])

        scaledFeatures: np.ndarray = self.pointsSwap * scalingWeights

        dbscan: DBSCAN = DBSCAN(eps=CLUSTER_EPS, min_samples=CLUSTER_MIN_SAMPLES)
        labels = dbscan.fit_predict(scaledFeatures)

        valid_mask = labels != -1
        self.pointsSwap = self.pointsSwap[valid_mask]
        self.pointsSwapLabels = labels[valid_mask]

    def visualizeClusteredStep(self) -> None:
        if self.dataFrame.empty:
            return

        fig: plt.Figure = plt.figure(figsize=(FIG_SIZE_X, FIG_SIZE_Y))
        ax = fig.add_subplot(111, projection='3d')
        
        
        midX: float = (self.minX + self.maxX) / 2
        groundZ: float = 0.0

        ax.plot([self.minX, self.minX], [self.minY, self.maxY], [groundZ, groundZ], color=LINE_COLOR, linewidth=LINE_WIDTH_VIS, label="Left Boundary")
        ax.plot([self.maxX, self.maxX], [self.minY, self.maxY], [groundZ, groundZ], color=LINE_COLOR, linewidth=LINE_WIDTH_VIS, label="Right Boundary")
        ax.plot([midX, midX], [self.minY, self.maxY], [groundZ, groundZ], color=CENTERLINE_COLOR, linestyle="--", linewidth=LINE_WIDTH_VIS, label="Centerline")

        if self.pointsSwap.size != 0:
            uniqueClusters: np.ndarray = np.unique(self.pointsSwapLabels)
            colors = plt.cm.get_cmap(COLOR_MAP, max(len(uniqueClusters), 1))

            for i, clusterId in enumerate(uniqueClusters):
                clusterMask = self.pointsSwapLabels == clusterId
                clusterData: np.ndarray = self.pointsSwap[clusterMask]

                ax.scatter(
                    clusterData[:, 0],
                    clusterData[:, 1],
                    clusterData[:, 2],
                    s=POINT_SIZE,
                    alpha=OPACITY_LEVEL,
                    color=colors(i),
                    label=f"Cluster {clusterId}"
                )

        ax.set_xlim(self.minX - 1, self.maxX + 1)
        ax.set_ylim(self.minY - 5, self.maxY + 5)
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

    def visualize(self) -> None:
        if self.dataFrame.empty:
            print("No radar data to visualize.")
            return

        self.findLane()
        minZ = float(self.dataFrame[Z_COLUMN].min())
        maxZ = float(self.dataFrame[Z_COLUMN].max())
        z_range = maxZ - minZ if maxZ != minZ else 1.0

        fig = plt.figure(figsize=(FIG_SIZE_X, FIG_SIZE_Y))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(
            self.dataFrame[X_COLUMN], 
            self.dataFrame[Y_COLUMN], 
            self.dataFrame[Z_COLUMN],
            s=POINT_SIZE / 4, 
            alpha=0.4,
            color='royalblue',
            label="Cloud of points"
        )

        midX = (self.minX + self.maxX) / 2
        groundZ = 0.0

        ax.plot([self.minX, self.minX], [self.minY, self.maxY], [groundZ, groundZ], 
                color='red', linewidth=LINE_WIDTH_VIS, label="Lanes")
        ax.plot([self.maxX, self.maxX], [self.minY, self.maxY], [groundZ, groundZ], 
                color='red', linewidth=LINE_WIDTH_VIS)

        ax.plot([midX, midX], [self.minY, self.maxY], [groundZ, groundZ], 
                color='red', linestyle="--", linewidth=1, alpha=0.7)
        x_span = self.maxX - self.minX
        y_span = self.maxY - self.minY
        ax.set_box_aspect((x_span, y_span, z_range))
        ax.set_xlim(self.minX, self.maxX)
        ax.set_ylim(self.minY, self.maxY)
        ax.set_zlim(minZ, maxZ)
        
        ax.set_xlabel("X (Width) [m]")
        ax.set_ylabel("Y (Distance) [m]")
        ax.set_zlabel("Z (Height) [m]")
        
        plt.legend()
        plt.tight_layout()
        plt.show(block=True)

    def findLane(self): 
        if self.dataFrame.empty:
            return {"minX": 0.0, "maxX": 0.0, "minY": 0.0, "maxY": 0.0}

        minX: float = float(self.dataFrame[X_COLUMN].min() - CORNER_OFFSET)
        maxX: float = float(self.dataFrame[X_COLUMN].max() + CORNER_OFFSET)
        minY: float = float(self.dataFrame[Y_COLUMN].min())
        maxY: float = float(self.dataFrame[Y_COLUMN].max())

        self.minX = minX
        self.maxX = maxX
        self.minY = minY
        self.maxY = maxY

        self.lane_width_meters = maxX - minX

    def calculateYaw(self, closeWindow: float = 20.0, farWindow: float = 20.0) -> float:
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

        deltaX: float = float(farPoints[COLUMN_X].median() - closePoints[COLUMN_X].median())
        deltaY: float = float(farPoints[COLUMN_Y].median() - closePoints[COLUMN_Y].median())

        return float(np.degrees(np.arctan2(deltaX, deltaY)))

    def adjustPoints(self, pitch: float, yaw: float, roll: float, heightOffset: float) -> None:
        pitchRad: float = np.radians(-pitch)
        yawRad: float = np.radians(-self.calculateYaw())
        rollRad: float = np.radians(-self.calculateRoll())
        if self.debug:
            print(f"Calculated Yaw: {np.degrees(yawRad):.2f} degrees, Calculated Roll: {np.degrees(rollRad):.2f} degrees")
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
        self._refresh_cache()

    def applyMask(self, zMin: float, zMax: float, yMin: float, yMax: float) -> None:
        self.dataFrame = self.dataFrame[
            (self.dataFrame[Y_COLUMN] >= yMin) &
            (self.dataFrame[Y_COLUMN] <= yMax) &
            (self.dataFrame[COLUMN_VELOCITY] != 0)
        ].copy()
        
        if not self.dataFrame.empty:
            self.currentTime = self.t0
        self._refresh_cache()

    def addNoise(self) -> None:
        if self.dataFrame.empty:
            return

        x = self.dataFrame[X_COLUMN].values
        y = self.dataFrame[Y_COLUMN].values
        z = self.dataFrame[Z_COLUMN].values
        v = self.dataFrame[COLUMN_VELOCITY].values

        dx = x
        dy = y
        dz = z - CAMERA_HEIGHT_OFFSET

        dist = np.sqrt(dx**2 + dy**2 + dz**2)
        azimuth = np.arctan2(dx, dy)
        elevation = np.arctan2(dz, np.sqrt(dx**2 + dy**2))

        n_dist = dist + np.random.uniform(-MAX_DISTANCE_ERROR, MAX_DISTANCE_ERROR, size=len(dist))
        n_az = azimuth + np.radians(np.random.uniform(-MAX_AZIMUTH_ERROR_DEG, MAX_AZIMUTH_ERROR_DEG, size=len(azimuth)))
        n_el = elevation + np.radians(np.random.uniform(-MAX_ELEVATION_ERROR_DEG, MAX_ELEVATION_ERROR_DEG, size=len(elevation)))

        cos_el = np.cos(n_el)
        
        self.dataFrame[X_COLUMN] = n_dist * cos_el * np.sin(n_az)
        self.dataFrame[Y_COLUMN] = n_dist * cos_el * np.cos(n_az)
        self.dataFrame[Z_COLUMN] = (n_dist * np.sin(n_el)) + CAMERA_HEIGHT_OFFSET

        if MAX_VELOCITY_ERROR > 0:
            self.dataFrame[COLUMN_VELOCITY] = v + np.random.uniform(-MAX_VELOCITY_ERROR, MAX_VELOCITY_ERROR, size=len(v))
        self._refresh_cache()

    def getClusterCenters(self) -> List[Dict[str, float]]:
        if self.pointsSwap.size == 0:
            return []

        centers: List[Dict[str, float]] = []
        for clusterId in np.unique(self.pointsSwapLabels):
            mask = self.pointsSwapLabels == clusterId
            clusterData = self.pointsSwap[mask]
            if clusterData.size == 0:
                continue
            centers.append({
                X_COLUMN: float(clusterData[:, 0].mean()),
                Y_COLUMN: float(clusterData[:, 1].mean()),
                Z_COLUMN: float(clusterData[:, 2].mean()),
                COLUMN_VELOCITY: float(clusterData[:, 3].mean() * -1),
            })

        self.clusterCenters = centers
        return self.clusterCenters
