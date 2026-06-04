import os
import numpy as np
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from typing import List, Dict

RELATIVE_CSV_PATH = "..\\..\\data\\normal_traffic\\radar_points_world.csv"
COLUMN_X = "x_world"    # lateral offset  (-1.7 to +5.3 m)
COLUMN_Y = "z_world"    # forward distance (6 to 82 m)
COLUMN_Z = "h_world"    # height above ground (3.6 to 10.9 m) — unnamed 12th CSV column
COLUMN_VELOCITY = "radial_velocity"
COLUMN_TIME = "timestamp"

SENSOR_PITCH_DEG = 0.0
SENSOR_YAW_DEG = 0.0
SENSOR_ROLL_DEG = 0.0
CAMERA_HEIGHT_OFFSET = 0.0  # world-frame coords already include absolute height

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

# Internal column indices for the data array (_data: N x 8)
_CI_X     = 0  # x_sensor
_CI_Y     = 1  # y_sensor
_CI_Z     = 2  # z_sensor
_CI_VEL   = 3  # radial_velocity
_CI_TIME  = 4  # timestamp
_CI_XCORR = 5  # x_corrected
_CI_YCORR = 6  # y_corrected
_CI_ZCORR = 7  # z_corrected
_NUM_COLS = 8


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
        self._data: np.ndarray = np.empty((0, _NUM_COLS), dtype=float)
        self.debug: bool = False
        self.loadData()
        self.adjustPoints(SENSOR_PITCH_DEG, SENSOR_YAW_DEG, SENSOR_ROLL_DEG, CAMERA_HEIGHT_OFFSET)
        self._refresh_cache()

    @staticmethod
    def _detect_world_columns(raw: np.ndarray, col_names: list) -> tuple:
        """
        Auto-detect which columns carry lateral X, forward Y and height Z.

        Criteria (per-column statistics across all rows):
          lateral X : has both negative and positive values (vmin < -0.5, vmax > 0.5)
                      AND range is road-width-scale: 2 m ≤ range ≤ 25 m
                      → among candidates pick the narrowest range (= actual road width)
          forward Y : strictly positive start (2 ≤ vmin ≤ 50 m)
                      AND plausible max distance (20 ≤ vmax ≤ 150 m)
                      AND large range (> 20 m)
                      → among candidates pick the one with the smallest vmin
                        (= closest to the sensor = actual depth, not world coords)
          height  Z : non-negative (vmin ≥ 0)
                      AND small absolute values (2 ≤ vmax ≤ 15 m)
                      AND meaningful spread (1 ≤ range ≤ 12 m)
                      AND column differs from forward Y
                      → pick narrowest range
        """
        lat_cands, fwd_cands, hgt_cands = [], [], []

        for i in range(raw.shape[1]):
            v = raw[:, i]
            vmin, vmax = float(v.min()), float(v.max())
            vrange = vmax - vmin

            if vmin < -0.5 and vmax > 0.5 and 2.0 <= vrange <= 25.0:
                lat_cands.append((i, vrange))
            if 2.0 <= vmin <= 50.0 and 20.0 <= vmax <= 150.0 and vrange > 20.0:
                fwd_cands.append((i, vmin))
            if vmin >= 0.0 and 2.0 <= vmax <= 15.0 and 1.0 <= vrange <= 12.0:
                hgt_cands.append((i, vrange))

        if not lat_cands or not fwd_cands or not hgt_cands:
            raise ValueError(
                "Cannot auto-detect lateral/forward/height columns. "
                f"Candidates — lat:{lat_cands} fwd:{fwd_cands} hgt:{hgt_cands}"
            )

        ci_x = min(lat_cands, key=lambda t: t[1])[0]   # narrowest → road width
        ci_y = min(fwd_cands, key=lambda t: t[1])[0]   # smallest min → closest to sensor
        hgt_f = [(i, r) for i, r in hgt_cands if i != ci_y]
        ci_z = min(hgt_f or hgt_cands, key=lambda t: t[1])[0]

        return ci_x, ci_y, ci_z

    def loadData(self) -> None:
        basePath: str = os.path.dirname(os.path.abspath(__file__))
        self.csvPath: str = os.path.abspath(os.path.join(basePath, self.relativePath))
        if not os.path.exists(self.csvPath):
            raise FileNotFoundError(f"File not found: {self.csvPath}")

        with open(self.csvPath, 'r') as f:
            header_line = f.readline().strip()
        col_names = [c.strip() for c in header_line.split(',')]

        raw: np.ndarray = np.genfromtxt(self.csvPath, delimiter=',', skip_header=1, dtype=float)
        if raw.ndim == 1:
            raw = raw.reshape(1, -1)

        # If CSV has more data columns than header names, the extra column is
        # an unnamed height column — label it "h_world".
        while raw.shape[1] > len(col_names):
            col_names.append("h_world")

        col_map = {name: i for i, name in enumerate(col_names)}

        # Sort by timestamp
        sort_idx = np.argsort(raw[:, col_map[COLUMN_TIME]], kind='stable')
        raw = raw[sort_idx]

        # Auto-detect the correct world-frame X/Y/Z columns regardless of CSV schema
        ci_x_src, ci_y_src, ci_z_src = self._detect_world_columns(raw, col_names)
        if self.debug:
            print(f"[Radar] Auto-detected columns — X:{col_names[ci_x_src]}  "
                  f"Y:{col_names[ci_y_src]}  Z:{col_names[ci_z_src]}")

        n = raw.shape[0]
        self._data = np.zeros((n, _NUM_COLS), dtype=float)
        self._data[:, _CI_X]    = raw[:, ci_x_src]
        self._data[:, _CI_Y]    = raw[:, ci_y_src]
        self._data[:, _CI_Z]    = raw[:, ci_z_src]

        # Velocity: if raw values exceed 1 000, assume mm/s → convert to m/s
        vel_raw = raw[:, col_map[COLUMN_VELOCITY]]
        vel_scale = 1000.0 if np.abs(vel_raw).max() > 1000.0 else 1.0
        self._data[:, _CI_VEL]  = vel_raw / vel_scale

        self._data[:, _CI_TIME] = raw[:, col_map[COLUMN_TIME]]
        # Corrected columns initialised to sensor values; overwritten by adjustPoints
        self._data[:, _CI_XCORR] = self._data[:, _CI_X]
        self._data[:, _CI_YCORR] = self._data[:, _CI_Y]
        self._data[:, _CI_ZCORR] = self._data[:, _CI_Z]

    def _refresh_cache(self) -> None:
        if self._data.shape[0] == 0:
            self._timeValues = np.empty((0,), dtype=float)
            self._pointsValues = np.empty((0, 4), dtype=float)
            return
        self._timeValues = self._data[:, _CI_TIME]
        self._pointsValues = self._data[:, [_CI_XCORR, _CI_YCORR, _CI_ZCORR, _CI_VEL]]

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
        if self._data.shape[0] == 0:
            return

        fig: plt.Figure = plt.figure(figsize=(FIG_SIZE_X, FIG_SIZE_Y))
        ax = fig.add_subplot(111, projection='3d')

        midX: float = (self.minX + self.maxX) / 2
        groundZ: float = float(self._data[:, _CI_ZCORR].min())

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

        zMin = float(self._data[:, _CI_ZCORR].min())
        zMax = float(self._data[:, _CI_ZCORR].max())
        ax.set_xlim(self.minX - 1, self.maxX + 1)
        ax.set_ylim(self.minY - 5, self.maxY + 5)
        ax.set_zlim(zMin, zMax)

        ax.set_xlabel("X (Width)")
        ax.set_ylabel("Y (Distance)")
        ax.set_zlabel("Z (Height)")
        ax.set_title(f"3D Radar Scene (T={self.currentTime:.2f}s)")
        ax.legend(loc='upper left', bbox_to_anchor=(1.05, 1))

        plt.tight_layout()
        plt.show(block=True)

    def calculateRoll(self, closeWindow: float = 20.0, farWindow: float = 20.0, numLowest: int = 10) -> float:
        if self._data.shape[0] == 0:
            return 0.0
        y = self._data[:, _CI_Y]
        yMin: float = float(y.min())
        yMax: float = float(y.max())
        closeMask = (y >= yMin) & (y <= yMin + closeWindow)
        farMask = (y >= yMax - farWindow) & (y <= yMax)
        closePoints = self._data[closeMask]
        farPoints = self._data[farMask]
        if closePoints.shape[0] == 0 or farPoints.shape[0] == 0:
            return 0.0
        closeLowest = closePoints[np.argsort(closePoints[:, _CI_Z])[:numLowest]]
        farLowest = farPoints[np.argsort(farPoints[:, _CI_Z])[:numLowest]]
        deltaZ: float = float(farLowest[:, _CI_Z].mean() - closeLowest[:, _CI_Z].mean())
        deltaY: float = float(farLowest[:, _CI_Y].mean() - closeLowest[:, _CI_Y].mean())
        return float(np.degrees(np.arctan2(deltaZ, deltaY)))

    def visualize(self) -> None:
        if self._data.shape[0] == 0:
            print("No radar data to visualize.")
            return

        self.findLane()
        minZ = float(self._data[:, _CI_ZCORR].min())
        maxZ = float(self._data[:, _CI_ZCORR].max())
        z_range = maxZ - minZ if maxZ != minZ else 1.0

        fig = plt.figure(figsize=(FIG_SIZE_X, FIG_SIZE_Y))
        ax = fig.add_subplot(111, projection='3d')

        ax.scatter(
            self._data[:, _CI_XCORR],
            self._data[:, _CI_YCORR],
            self._data[:, _CI_ZCORR],
            s=POINT_SIZE / 4,
            alpha=0.4,
            color='royalblue',
            label="Cloud of points"
        )

        midX = (self.minX + self.maxX) / 2
        groundZ = minZ  # draw lane lines at the lowest Z in the dataset

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
        if self._data.shape[0] == 0:
            return {"minX": 0.0, "maxX": 0.0, "minY": 0.0, "maxY": 0.0}

        minX: float = float(self._data[:, _CI_XCORR].min()) - CORNER_OFFSET
        maxX: float = float(self._data[:, _CI_XCORR].max()) + CORNER_OFFSET
        minY: float = float(self._data[:, _CI_YCORR].min())
        maxY: float = float(self._data[:, _CI_YCORR].max())

        self.minX = minX
        self.maxX = maxX
        self.minY = minY
        self.maxY = maxY

        self.lane_width_meters = maxX - minX

    def calculateYaw(self, closeWindow: float = 20.0, farWindow: float = 20.0) -> float:
        if self._data.shape[0] == 0:
            return 0.0

        y = self._data[:, _CI_Y]
        yMin: float = float(y.min())
        yMax: float = float(y.max())

        closeMask = (y >= yMin) & (y <= yMin + closeWindow)
        farMask = (y >= yMax - farWindow) & (y <= yMax)

        closePoints = self._data[closeMask]
        farPoints = self._data[farMask]

        if closePoints.shape[0] == 0 or farPoints.shape[0] == 0:
            return 0.0

        deltaX: float = float(np.median(farPoints[:, _CI_X]) - np.median(closePoints[:, _CI_X]))
        deltaY: float = float(np.median(farPoints[:, _CI_Y]) - np.median(closePoints[:, _CI_Y]))

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
        x = self._data[:, _CI_X]
        y = self._data[:, _CI_Y]
        z = self._data[:, _CI_Z]
        x1 = x * cosY + y * sinY
        y1 = -x * sinY + y * cosY
        x2 = x1 * cosP + z * sinP
        z2 = -x1 * sinP + z * cosP
        self._data[:, _CI_XCORR] = x2
        self._data[:, _CI_YCORR] = y1 * cosR - z2 * sinR
        self._data[:, _CI_ZCORR] = (y1 * sinR + z2 * cosR) + heightOffset
        self._refresh_cache()

    def applyMask(self, zMin: float, zMax: float, yMin: float, yMax: float) -> None:
        # Keep points within the forward-distance window.
        # Velocity filter: only drop points whose |velocity| is negligible
        # (< 0.1 m/s) AND the dataset has *some* moving points.  This avoids
        # discarding all data in low-speed / same-speed scenarios (e.g. 1_Control).
        vel = self._data[:, _CI_VEL]
        moving_fraction = (np.abs(vel) > 0.1).mean()
        if moving_fraction > 0.05:
            vel_mask = np.abs(vel) > 0.1
        else:
            vel_mask = np.ones(len(vel), dtype=bool)  # keep everything

        mask = (
            (self._data[:, _CI_YCORR] >= yMin) &
            (self._data[:, _CI_YCORR] <= yMax) &
            vel_mask
        )
        self._data = self._data[mask].copy()

        if self._data.shape[0] > 0:
            self.currentTime = self.t0
        self._refresh_cache()

    def addNoise(self) -> None:
        if self._data.shape[0] == 0:
            return

        x = self._data[:, _CI_XCORR].copy()
        y = self._data[:, _CI_YCORR].copy()
        z = self._data[:, _CI_ZCORR].copy()
        v = self._data[:, _CI_VEL].copy()

        dz = z - CAMERA_HEIGHT_OFFSET
        dist = np.sqrt(x**2 + y**2 + dz**2)
        azimuth = np.arctan2(x, y)
        elevation = np.arctan2(dz, np.sqrt(x**2 + y**2))

        n = len(dist)
        n_dist = dist + np.random.uniform(-MAX_DISTANCE_ERROR, MAX_DISTANCE_ERROR, size=n)
        n_az   = azimuth + np.radians(np.random.uniform(-MAX_AZIMUTH_ERROR_DEG, MAX_AZIMUTH_ERROR_DEG, size=n))
        n_el   = elevation + np.radians(np.random.uniform(-MAX_ELEVATION_ERROR_DEG, MAX_ELEVATION_ERROR_DEG, size=n))

        cos_el = np.cos(n_el)
        self._data[:, _CI_XCORR] = n_dist * cos_el * np.sin(n_az)
        self._data[:, _CI_YCORR] = n_dist * cos_el * np.cos(n_az)
        self._data[:, _CI_ZCORR] = (n_dist * np.sin(n_el)) + CAMERA_HEIGHT_OFFSET

        if MAX_VELOCITY_ERROR > 0:
            self._data[:, _CI_VEL] = v + np.random.uniform(-MAX_VELOCITY_ERROR, MAX_VELOCITY_ERROR, size=n)
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