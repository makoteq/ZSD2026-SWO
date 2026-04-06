import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from sklearn.linear_model import LinearRegression
import os

RELATIVE_CSV_PATH: str = "../../data\\batch2\\batch2\\scenario1_speeding_2cars\\run_003\\radar_points_world.csv"

COLUMN_X: str = "x_sensor"
COLUMN_Y: str = "y_sensor"
COLUMN_Z: str = "z_sensor"

SENSOR_PITCH_DEG: float = 0.0
SENSOR_YAW_DEG: float = 0.0
SENSOR_ROLL_DEG: float = 0.0
CAMERA_HEIGHT_OFFSET: float = 6.0

MASK_Z_MIN: float = 0.15
MASK_Z_MAX: float = 5.0
MASK_Y_MIN: float = 72.0
MASK_Y_MAX: float = 90.0

GRID_RES: float = 0.1
STREET_Z_THRESHOLD: float = 0.5
MORPH_ITERATIONS: int = 3
MIN_CONTOUR_POINTS: int = 15
INTERPOLATION_FACTOR: int = 2
SIZE_POINT_PHOTO: int = 700

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

    def calculateRoll(self, closeWindow: float = 20.0, farWindow: float = 20.0, numLowest: int = 10) -> float:
        if self.dataFrame.empty:
            return 0.0
            
        y: pd.Series = self.dataFrame[COLUMN_Y]
        
        yMin: float = y.min()
        yMax: float = y.max()
        
        closeMask = (y >= yMin) & (y <= yMin + closeWindow)
        farMask = (y >= yMax - farWindow) & (y <= yMax)
        
        closePoints = self.dataFrame[closeMask]
        farPoints = self.dataFrame[farMask]
        
        if closePoints.empty or farPoints.empty:
            return 0.0

        closeLowest = closePoints.nsmallest(numLowest, COLUMN_Z)
        farLowest = farPoints.nsmallest(numLowest, COLUMN_Z)
        
        zCloseMean: float = closeLowest[COLUMN_Z].mean()
        yCloseMean: float = closeLowest[COLUMN_Y].mean()
        
        zFarMean: float = farLowest[COLUMN_Z].mean()
        yFarMean: float = farLowest[COLUMN_Y].mean()
        
        deltaZ: float = zFarMean - zCloseMean
        deltaY: float = yFarMean - yCloseMean
        
        angleRad: float = np.arctan2(deltaZ, deltaY)

        return float(np.degrees(angleRad))

    def adjustPoints(self, pitch: float, yaw: float, roll: float, heightOffset: float) -> None:
        pitchRad: float = np.radians(-pitch)
        yawRad: float = np.radians(-yaw)
        roll = self.calculateRoll()
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
            (self.dataFrame["y_corrected"] <= yMax) &
            (self.dataFrame["radial_velocity"] == 0)
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

    def findContours(self) -> None:
        if self.dataFrame.empty:
            return

        x, y, z = self.dataFrame["x_corrected"].values, self.dataFrame["y_corrected"].values, self.dataFrame["z_corrected"].values
        xMin, xMax, yMin, yMax = x.min(), x.max(), y.min(), y.max()
        w, h = int((yMax - yMin) / GRID_RES) + 1, int((xMax - xMin) / GRID_RES) + 1

        hMap = np.zeros((h, w), dtype=np.float32)
        yIdx, xIdx = ((y - yMin) / GRID_RES).astype(int), ((x - xMin) / GRID_RES).astype(int)
        for i in range(len(z)):
            if hMap[xIdx[i], yIdx[i]] < z[i]:
                hMap[xIdx[i], yIdx[i]] = z[i]

  

        fig = plt.figure(frameon=False)
        fig.set_size_inches(10, 8)
        ax = plt.Axes(fig, [0., 0., 1., 1.])
        ax.set_axis_off()
        fig.add_axes(ax)
        
        ax.imshow(hMap, cmap='plasma', origin='lower', extent=[xMin, xMax, yMin, yMax], alpha=0, aspect='auto')
        ax.scatter(x, y, c=z, s=SIZE_POINT_PHOTO, cmap='plasma', edgecolors='k', linewidth=0)
        
        fig.savefig('bird_eye_heatmap.png', dpi=300)
        # plt.show()


    def enhanceImage(self) -> None:
        img = cv2.imread('bird_eye_heatmap.png')
        if img is None:
            print("Image 'bird_eye_heatmap.png' not found.")
            return
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        
        kernel = 101
        sigma = 50
        contrast = 1.75
        
        if kernel > 1:
            blurred = cv2.GaussianBlur(img, (kernel, kernel), sigma)
        else:
            blurred = img
        enhanced = cv2.convertScaleAbs(blurred, alpha=contrast, beta=0)
        
        sharpen_kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
        sharpened = cv2.filter2D(enhanced, -1, sharpen_kernel)
        
        ax.imshow(sharpened)
        ax.set_title('Enhanced and Sharpened Image')
        
        plt.savefig('enhanced_sharpened.png', dpi=300, bbox_inches='tight')
        plt.show()

if __name__ == "__main__":
    streetInstance = Street(RELATIVE_CSV_PATH)
    streetInstance.adjustPoints(SENSOR_PITCH_DEG, SENSOR_YAW_DEG, SENSOR_ROLL_DEG, CAMERA_HEIGHT_OFFSET)
    streetInstance.applyMask(MASK_Z_MIN, MASK_Z_MAX, MASK_Y_MIN, MASK_Y_MAX)
    streetInstance.interpolateCloud(INTERPOLATION_FACTOR)

    streetInstance.findContours()
    streetInstance.enhanceImage()
    # streetInstance.visualize()