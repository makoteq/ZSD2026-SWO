import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from sklearn.linear_model import LinearRegression
import os

RELATIVE_CSV_PATH: str = "../../data/carla/scenario5_empty_road/radar_points_world.csv"

COLUMN_X: str = "x_sensor"
COLUMN_Y: str = "y_sensor"
COLUMN_Z: str = "z_sensor"

SENSOR_PITCH_DEG: float = 0.0
SENSOR_YAW_DEG: float = 0.0
SENSOR_ROLL_DEG: float = 5.0
CAMERA_HEIGHT_OFFSET: float = 6.0

MASK_Z_MIN: float = 0.0
MASK_Z_MAX: float = 1.0
MASK_Y_MIN: float = 00.0
MASK_Y_MAX: float = 200.0

GRID_RES: float = 0.1
STREET_Z_THRESHOLD: float = 0.5
MORPH_ITERATIONS: int = 3
MIN_CONTOUR_POINTS: int = 15
INTERPOLATION_FACTOR: int = 2

SIGMOID_STEEPNESS: float = 31.94
SIGMOID_OFFSET: float = 0.0714

POINT_SIZE: int = 1
OPACITY_LEVEL: float = 1.0
PLOT_TITLE: str = "Radar Street Detection - Dense Cloud"
RENDERER_TYPE: str = "browser"

class Street:
    def __init__(self, relativePath: str) -> None:
        basePath: str = os.path.dirname(os.path.abspath(__file__))
        self.csvPath: str = os.path.abspath(os.path.join(basePath, relativePath))
        
        if not os.path.exists(self.csvPath):
            raise FileNotFoundError(f"File not found: {self.csvPath}")
            
        self.dataFrame: pd.DataFrame = pd.read_csv(self.csvPath)
        self.contours: list[np.ndarray] = []
        self.fittedLines: list[np.ndarray] = []
        self.lastUsedOffset: float = SIGMOID_OFFSET

    def adjustPoints(self, pitch: float, yaw: float, roll: float, heightOffset: float) -> None:
        pitchRad: float = np.radians(-pitch)
        yawRad: float = np.radians(-yaw)
        rollRad: float = np.radians(-roll)

        cosP, sinP = np.cos(pitchRad), np.sin(pitchRad)
        cosY, sinY = np.cos(yawRad), np.sin(yawRad)
        cosR, sinR = np.cos(rollRad), np.sin(rollRad)

        x: pd.Series = self.dataFrame[COLUMN_X]
        y: pd.Series = self.dataFrame[COLUMN_Y]
        z: pd.Series = self.dataFrame[COLUMN_Z]

        x1: pd.Series = x * cosY + y * sinY
        y1: pd.Series = -x * sinY + y * cosY
        z1: pd.Series = z

        x2: pd.Series = x1 * cosP + z1 * sinP
        y2: pd.Series = y1
        z2: pd.Series = -x1 * sinP + z1 * cosP

        self.dataFrame["x_corrected"] = x2
        self.dataFrame["y_corrected"] = y2 * cosR - z2 * sinR
        self.dataFrame["z_corrected"] = (y2 * sinR + z2 * cosR) + heightOffset

    def applyMask(self, zMin: float, zMax: float, yMin: float, yMax: float) -> None:
        self.dataFrame = self.dataFrame[
            (self.dataFrame["z_corrected"] >= zMin) & 
            (self.dataFrame["z_corrected"] <= zMax) &
            (self.dataFrame["y_corrected"] >= yMin) &
            (self.dataFrame["y_corrected"] <= yMax)
        ].copy()

    def interpolateCloud(self, factor: int) -> None:
        if self.dataFrame.empty or factor <= 1:
            return
        
        x: np.ndarray = self.dataFrame["x_corrected"].values
        y: np.ndarray = self.dataFrame["y_corrected"].values
        z: np.ndarray = self.dataFrame["z_corrected"].values
        
        interpX: list[float] = []
        interpY: list[float] = []
        interpZ: list[float] = []
        
        for i in range(len(x) - 1):
            interpX.extend(np.linspace(x[i], x[i+1], factor))
            interpY.extend(np.linspace(y[i], y[i+1], factor))
            interpZ.extend(np.linspace(z[i], z[i+1], factor))
            
        self.dataFrame = pd.DataFrame({
            "x_corrected": interpX,
            "y_corrected": interpY,
            "z_corrected": interpZ
        })

    def _sigmoid(self, zValues: np.ndarray, steep: float, off: float) -> np.ndarray:
        return 1 / (1 + np.exp(-steep * (zValues - off)))

    def tuneSigmoid(self) -> None:
        if self.dataFrame.empty:
            return

        x, y, z = self.dataFrame["x_corrected"].values, self.dataFrame["y_corrected"].values, self.dataFrame["z_corrected"].values
        xMin, xMax, yMin, yMax = x.min(), x.max(), y.min(), y.max()
        w, h = int((yMax - yMin) / GRID_RES) + 1, int((xMax - xMin) / GRID_RES) + 1
        
        self.tuneHMap = np.zeros((h, w), dtype=np.float32)
        yIdx, xIdx = ((y - yMin) / GRID_RES).astype(int), ((x - xMin) / GRID_RES).astype(int)
        for i in range(len(z)):
            if self.tuneHMap[xIdx[i], yIdx[i]] < z[i]:
                self.tuneHMap[xIdx[i], yIdx[i]] = z[i]

        self.fig, self.ax = plt.subplots(1, 2, figsize=(14, 7))
        plt.subplots_adjust(bottom=0.25)

        normInit = (self._sigmoid(self.tuneHMap, SIGMOID_STEEPNESS, SIGMOID_OFFSET) * 255).astype(np.uint8)
        self.imgDisp = self.ax[0].imshow(normInit, cmap='hot', origin='lower')
        
        _, binInit = cv2.threshold(normInit, int(STREET_Z_THRESHOLD * 255), 255, cv2.THRESH_BINARY)
        self.imgBin = self.ax[1].imshow(binInit, cmap='gray', origin='lower')

        axOff = plt.axes([0.2, 0.1, 0.6, 0.03])
        axStp = plt.axes([0.2, 0.05, 0.6, 0.03])
        self.sliderOff = Slider(axOff, 'Offset (Z)', 0.0, 0.5, valinit=SIGMOID_OFFSET)
        self.sliderStp = Slider(axStp, 'Steepness', 1.0, 50.0, valinit=SIGMOID_STEEPNESS)

        def update(val):
            enhanced = self._sigmoid(self.tuneHMap, self.sliderStp.val, self.sliderOff.val)
            norm = (enhanced * 255).astype(np.uint8)
            self.imgDisp.set_data(norm)
            _, binary = cv2.threshold(norm, int(STREET_Z_THRESHOLD * 255), 255, cv2.THRESH_BINARY)
            self.imgBin.set_data(binary)
            self.fig.canvas.draw_idle()

        self.sliderOff.on_changed(update)
        self.sliderStp.on_changed(update)
        plt.show()

    def findStreetContours(self, steepness: float = SIGMOID_STEEPNESS, offset: float = SIGMOID_OFFSET) -> None:
        if self.dataFrame.empty:
            return

        self.lastUsedOffset = offset
        x, y, z = self.dataFrame["x_corrected"].values, self.dataFrame["y_corrected"].values, self.dataFrame["z_corrected"].values
        xMin, xMax, yMin, yMax = x.min(), x.max(), y.min(), y.max()
        w, h = int((yMax - yMin) / GRID_RES) + 1, int((xMax - xMin) / GRID_RES) + 1

        hMap = np.zeros((h, w), dtype=np.float32)
        yIdx, xIdx = ((y - yMin) / GRID_RES).astype(int), ((x - xMin) / GRID_RES).astype(int)
        for i in range(len(z)):
            if hMap[xIdx[i], yIdx[i]] < z[i]:
                hMap[xIdx[i], yIdx[i]] = z[i]

        enhanced = self._sigmoid(hMap, steepness, offset)
        normalized = (enhanced * 255).astype(np.uint8)
        
        _, binary = cv2.threshold(normalized, int(STREET_Z_THRESHOLD * 255), 255, cv2.THRESH_BINARY)
        kernel = np.ones((3, 3), np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=MORPH_ITERATIONS)
        
        rawContours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        self.contours, self.fittedLines = [], []
        for cnt in rawContours:
            pts = cnt.reshape(-1, 2)
            if len(pts) < MIN_CONTOUR_POINTS:
                continue
            
            wX = pts[:, 1] * GRID_RES + xMin
            wY = pts[:, 0] * GRID_RES + yMin

            model = LinearRegression()
            X_reg = wX.reshape(-1, 1)
            model.fit(X_reg, wY)
            
            self.contours.append(np.column_stack((wX, wY)))
            xRange = np.array([wX.min(), wX.max()]).reshape(-1, 1)
            self.fittedLines.append(np.column_stack((xRange, model.predict(xRange))))

    def visualize(self) -> None:
        fig = px.scatter_3d(
            self.dataFrame, x="x_corrected", y="y_corrected", z="z_corrected",
            color="z_corrected", color_continuous_scale="Turbo",
            title=PLOT_TITLE, opacity=OPACITY_LEVEL
        )
        for i, line in enumerate(self.fittedLines):
            fig.add_trace(go.Scatter3d(
                x=line[:, 0], y=line[:, 1], z=[self.lastUsedOffset, self.lastUsedOffset],
                mode='lines', line=dict(color='yellow', width=10), name=f'Edge {i}'
            ))
        fig.update_layout(scene=dict(aspectmode="data"), margin=dict(l=0, r=0, b=0, t=40))
        fig.show(renderer=RENDERER_TYPE)

if __name__ == "__main__":
    streetInstance = Street(RELATIVE_CSV_PATH)
    streetInstance.adjustPoints(SENSOR_PITCH_DEG, SENSOR_YAW_DEG, SENSOR_ROLL_DEG, CAMERA_HEIGHT_OFFSET)
    streetInstance.applyMask(MASK_Z_MIN, MASK_Z_MAX, MASK_Y_MIN, MASK_Y_MAX)
    

    streetInstance.interpolateCloud(INTERPOLATION_FACTOR)
    
    # streetInstance.findStreetContours()
    streetInstance.visualize()