import os
import gc
import csv
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from typing import List, Dict
import psutil

RELATIVE_CSV_PATH = "..\\..\\data\\normal_traffic\\radar_points_world.csv"
BASE_PATH = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.abspath(os.path.join(BASE_PATH, RELATIVE_CSV_PATH))

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
    def __init__(self, csv_path: str, start_time: float) -> None:
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
        self.dataFrame = None
        self.csvPath = csv_path
        self.outputCsvPath = os.path.join(
            os.path.dirname(self.csvPath), 
            os.path.basename(self.csvPath).replace(".csv", "_filtered.csv")
        )

        self.debug: bool = False
        self.loadData()
        self.adjustPoints(SENSOR_PITCH_DEG, SENSOR_YAW_DEG, SENSOR_ROLL_DEG, CAMERA_HEIGHT_OFFSET)

    def loadData(self) -> None:
        if not os.path.exists(self.csvPath):
            raise FileNotFoundError(f"File not found: {self.csvPath}")
        
        df = pd.read_csv(self.csvPath)
        df.sort_values(COLUMN_TIME, inplace=True)
        df.reset_index(drop=True, inplace=True)
        df.to_csv(self.outputCsvPath, index=False)
        
        del df
        self.dataFrame = None
        gc.collect()

    def step(self, timeStep: float) -> None:
        self.pointsSwap = np.empty((0, 4), dtype=float)
        self.pointsSwapLabels = np.empty((0,), dtype=int)
        
        endTime: float = self.currentTime + timeStep
        collected_points = []
        
        with open(self.outputCsvPath, mode='r', newline='', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            
            idx_time = header.index(COLUMN_TIME)
            idx_x = header.index(X_COLUMN)
            idx_y = header.index(Y_COLUMN)
            idx_z = header.index(Z_COLUMN)
            idx_v = header.index(COLUMN_VELOCITY)
            
            for row in reader:
                if not row:
                    continue
                t = float(row[idx_time])
                if self.currentTime <= t < endTime:
                    collected_points.append([
                        float(row[idx_x]),
                        float(row[idx_y]),
                        float(row[idx_z]),
                        float(row[idx_v])
                    ])
                elif t >= endTime:
                    break
                    
        if collected_points:
            self.pointsSwap = np.array(collected_points, dtype=float)
            
        self.currentTime = endTime
        del collected_points
        gc.collect()

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
        
        new_points = self.pointsSwap[valid_mask]
        new_labels = labels[valid_mask]
        
        self.pointsSwap = np.empty((0, 4), dtype=float)
        self.pointsSwapLabels = np.empty((0,), dtype=int)
        
        self.pointsSwap = new_points
        self.pointsSwapLabels = new_labels
        
        del scaledFeatures, labels, valid_mask
        gc.collect()

    def visualizeClusteredStep(self) -> None:
        fig: plt.Figure = plt.figure(figsize=(FIG_SIZE_X, FIG_SIZE_Y))
        ax = fig.add_subplot(111, projection='3d')
        
        midX: float = (self.minX + self.maxX) / 2
        groundZ: float = 0.0

        ax.plot([self.minX, self.minX], [self.minY, self.maxY], [groundZ, groundZ], color=LINE_COLOR, linewidth=LINE_WIDTH_VIS, label="Left Boundary")
        ax.plot([self.maxX, self.maxX], [self.minY, self.maxY], [groundZ, groundZ], color=LINE_COLOR, linewidth=LINE_WIDTH_VIS, label="Right Boundary")
        ax.plot([midX, midX], [self.minY, self.maxY], [groundZ, groundZ], color=CENTERLINE_COLOR, linestyle="--", linewidth=LINE_WIDTH_VIS, label="Centerline")

        if self.pointsSwap.size != 0:
            uniqueClusters: np.ndarray = np.unique(self.pointsSwapLabels)
            colors = matplotlib.colormaps[COLOR_MAP].resampled(max(len(uniqueClusters), 1))

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
        plt.close(fig)

    def calculateRoll(self, closeWindow: float = 20.0, farWindow: float = 20.0, numLowest: int = 10) -> float:
        df = pd.read_csv(self.outputCsvPath, usecols=[COLUMN_Y, COLUMN_Z])
        if df.empty:
            del df
            gc.collect()
            return 0.0
        y: pd.Series = df[COLUMN_Y]
        yMin: float = float(y.min())
        yMax: float = float(y.max())
        closeMask = (y >= yMin) & (y <= yMin + closeWindow)
        farMask = (y >= yMax - farWindow) & (y <= yMax)
        closePoints: pd.DataFrame = df[closeMask]
        farPoints: pd.DataFrame = df[farMask]
        if closePoints.empty or farPoints.empty:
            del df, closePoints, farPoints
            gc.collect()
            return 0.0
        closeLowest: pd.DataFrame = closePoints.nsmallest(numLowest, COLUMN_Z)
        farLowest: pd.DataFrame = farPoints.nsmallest(numLowest, COLUMN_Z)
        deltaZ: float = float(farLowest[COLUMN_Z].mean() - closeLowest[COLUMN_Z].mean())
        deltaY: float = float(farLowest[COLUMN_Y].mean() - closeLowest[COLUMN_Y].mean())
        
        del df, closePoints, farPoints, closeLowest, farLowest
        gc.collect()
        return float(np.degrees(np.arctan2(deltaZ, deltaY)))

    def visualize(self) -> None:
        self.findLane()
        df = pd.read_csv(self.outputCsvPath, usecols=[X_COLUMN, Y_COLUMN, Z_COLUMN])
        if df.empty:
            print("No radar data to visualize.")
            del df
            gc.collect()
            return

        minZ = float(df[Z_COLUMN].min())
        maxZ = float(df[Z_COLUMN].max())
        z_range = maxZ - minZ if maxZ != minZ else 1.0

        fig = plt.figure(figsize=(FIG_SIZE_X, FIG_SIZE_Y))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(
            df[X_COLUMN], 
            df[Y_COLUMN], 
            df[Z_COLUMN],
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
        plt.close(fig)
        del df
        gc.collect()

    def findLane(self): 
        df = pd.read_csv(self.outputCsvPath, usecols=[X_COLUMN, Y_COLUMN])
        if df.empty:
            del df
            gc.collect()
            return

        minX: float = float(df[X_COLUMN].min() - CORNER_OFFSET)
        maxX: float = float(df[X_COLUMN].max() + CORNER_OFFSET)
        minY: float = float(df[Y_COLUMN].min())
        maxY: float = float(df[Y_COLUMN].max())

        self.minX = minX
        self.maxX = maxX
        self.minY = minY
        self.maxY = maxY

        self.lane_width_meters = maxX - minX
        del df
        gc.collect()

    def calculateYaw(self, closeWindow: float = 20.0, farWindow: float = 20.0) -> float:
        df = pd.read_csv(self.outputCsvPath, usecols=[COLUMN_X, COLUMN_Y])
        if df.empty:
            del df
            gc.collect()
            return 0.0

        y: pd.Series = df[COLUMN_Y]
        yMin: float = float(y.min())
        yMax: float = float(y.max())

        closeMask = (y >= yMin) & (y <= yMin + closeWindow)
        farMask = (y >= yMax - farWindow) & (y <= yMax)

        closePoints: pd.DataFrame = df[closeMask]
        farPoints: pd.DataFrame = df[farMask]

        if closePoints.empty or farPoints.empty:
            del df, closePoints, farPoints
            gc.collect()
            return 0.0

        deltaX: float = float(farPoints[COLUMN_X].median() - closePoints[COLUMN_X].median())
        deltaY: float = float(farPoints[COLUMN_Y].median() - closePoints[COLUMN_Y].median())

        del df, closePoints, farPoints
        gc.collect()
        return float(np.degrees(np.arctan2(deltaX, deltaY)))

    def adjustPoints(self, pitch: float, yaw: float, roll: float, heightOffset: float) -> None:
        df = pd.read_csv(self.outputCsvPath)
        pitchRad: float = np.radians(-pitch)
        yawRad: float = np.radians(-self.calculateYaw())
        rollRad: float = np.radians(-self.calculateRoll())
        
        cosP, sinP = np.cos(pitchRad), np.sin(pitchRad)
        cosY, sinY = np.cos(yawRad), np.sin(yawRad)
        cosR, sinR = np.cos(rollRad), np.sin(rollRad)
        
        x: pd.Series = df[COLUMN_X]
        y: pd.Series = df[COLUMN_Y]
        z: pd.Series = df[COLUMN_Z]
        
        x1: pd.Series = x * cosY + y * sinY
        y1: pd.Series = -x * sinY + y * cosY
        x2: pd.Series = x1 * cosP + z * sinP
        z2: pd.Series = -x1 * sinP + z * cosP
        
        df[X_COLUMN] = x2
        df[Y_COLUMN] = y1 * cosR - z2 * sinR
        df[Z_COLUMN] = (y1 * sinR + z2 * cosR) + heightOffset
        df.to_csv(self.outputCsvPath, index=False)
        del df
        gc.collect()

    def applyMask(self, zMin: float, zMax: float, yMin: float, yMax: float) -> None:
        df = pd.read_csv(self.outputCsvPath)
        df = df[
            (df[Y_COLUMN] >= yMin) &
            (df[Y_COLUMN] <= yMax) &
            (df[COLUMN_VELOCITY] != 0)
        ].copy()
        
        if not df.empty:
            self.currentTime = self.t0
        df.to_csv(self.outputCsvPath, index=False)
        del df
        gc.collect()

    def addNoise(self) -> None:
        df = pd.read_csv(self.outputCsvPath)
        if df.empty:
            del df
            gc.collect()
            return

        x = df[X_COLUMN].values
        y = df[Y_COLUMN].values
        z = df[Z_COLUMN].values
        v = df[COLUMN_VELOCITY].values
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
        
        df[X_COLUMN] = n_dist * cos_el * np.sin(n_az)
        df[Y_COLUMN] = n_dist * cos_el * np.cos(n_az)
        df[Z_COLUMN] = (n_dist * np.sin(n_el)) + CAMERA_HEIGHT_OFFSET

        if MAX_VELOCITY_ERROR > 0:
            df[COLUMN_VELOCITY] = v + np.random.uniform(-MAX_VELOCITY_ERROR, MAX_VELOCITY_ERROR, size=len(v))
        df.to_csv(self.outputCsvPath, index=False)
        del df
        gc.collect()

    def getClusterCenters(self) -> List[Dict[str, float]]:
        if isinstance(self.clusterCenters, list):
            self.clusterCenters.clear()
        self.clusterCenters = []

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


if __name__ == "__main__":
    radar = Radar(csv_path=CSV_PATH, start_time=0.0)
    radar.debug = True

    radar.applyMask(MASK_Z_MIN, MASK_Z_MAX, MASK_Y_MIN, MASK_Y_MAX)
    radar.addNoise()
    radar.findLane()
    for _ in range(LOOP_ITERATIONS):
        radar.step(TIME_STEP_DEFAULT)
        radar.clusterPoints()
        radar.getClusterCenters()