import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import cv2
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from sklearn.linear_model import LinearRegression
import os

RELATIVE_CSV_PATH: str = "../../data\\batch2\\batch2\\scenario1_speeding_2cars\\run_004\\radar_points_world.csv"

COLUMN_X: str = "x_sensor"
COLUMN_Y: str = "y_sensor"
COLUMN_Z: str = "z_sensor"

SENSOR_PITCH_DEG: float = 0.0
SENSOR_YAW_DEG: float = 0.0
SENSOR_ROLL_DEG: float = 0.0
CAMERA_HEIGHT_OFFSET: float = 6.0

MASK_Z_MIN: float = 0.15
MASK_Z_MAX: float = 5.0
MASK_Y_MIN: float = 70.0
MASK_Y_MAX: float = 120.0


GRID_RES: float = 0.1
STREET_Z_THRESHOLD: float = 0.5
MORPH_ITERATIONS: int = 3
MIN_CONTOUR_POINTS: int = 15
INTERPOLATION_FACTOR: int = 2


SIGMOID_STEEPNESS: float = 31.94
SIGMOID_OFFSET: float = 0.0714

POINT_SIZE: int = 5
LINE_THICKNESS: int = 30
OPACITY_LEVEL: float = 1.0
PLOT_TITLE: str = "Radar Street Detection - Dense Cloud"
RENDERER_TYPE: str = "browser"

GAUSSIAN_KERNEL: int = 101
GAUSSIAN_SIGMA: int = 50
CONTRAST_ALPHA: float = 1.75

CANNY_THRESHOLD_1: int = 50
CANNY_THRESHOLD_2: int = 150
CONTOUR_COLOR: tuple = (255, 0, 0)
CONTOUR_THICKNESS: int = 3
NUM_CONTOURS: int = 2

class Street:
    def __init__(self, relativePath: str) -> None:
        basePath: str = os.path.dirname(os.path.abspath(__file__))
        self.csvPath: str = os.path.abspath(os.path.join(basePath, relativePath))
        
        if not os.path.exists(self.csvPath):
            raise FileNotFoundError(f"File not found: {self.csvPath}")
            
        self.dataFrame: pd.DataFrame = pd.read_csv(self.csvPath)
        
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

    def findRoad(self) -> np.ndarray:
        if self.dataFrame.empty:
            return np.array([])

        z = self.dataFrame["z_corrected"].values

     
        bin_size = 0.00656
        bins = np.arange(np.min(z), np.max(z) + bin_size, bin_size)
        counts, _ = np.histogram(z, bins=bins)
        bin_centers = (bins[:-1] + bins[1:]) / 2

        road_height = bin_centers[np.argmax(counts)]

        tolerance = 0.001
        road_mask = (z >= road_height - tolerance) & (z <= road_height + tolerance)
        
        self.dataFrame["is_road"] = road_mask
        
        return 

   

    def findContours(self) -> None:
            roadPoints: pd.DataFrame = self.dataFrame[self.dataFrame["is_road"] == True]

            if roadPoints.empty:
                return

            minX: float = float(roadPoints["x_corrected"].min())
            maxX: float = float(roadPoints["x_corrected"].max())
            minY: float = float(roadPoints["y_corrected"].min())
            maxY: float = float(roadPoints["y_corrected"].max())
            roadHeight: float = float(roadPoints["z_corrected"].mean())

            self.contours = [
                np.array([[minX, minY, roadHeight], [minX, maxY, roadHeight]]),
                np.array([[maxX, minY, roadHeight], [maxX, maxY, roadHeight]])
        ]

    def visualize(self) -> None:
            if "is_road" not in self.dataFrame.columns:
                self.dataFrame["is_road"] = False

            fig = px.scatter_3d(
                self.dataFrame, 
                x="x_corrected", 
                y="y_corrected", 
                z="z_corrected",
                color="z_corrected",
                color_continuous_scale="Turbo",
                title=PLOT_TITLE, 
                opacity=OPACITY_LEVEL
            )

            fig.update_traces(marker=dict(size=POINT_SIZE))

            for i, line in enumerate(self.contours):
                fig.add_trace(go.Scatter3d(
                    x=line[:, 0], 
                    y=line[:, 1], 
                    z=line[:, 2],
                    mode='lines', 
                    line=dict(color='green', width=LINE_THICKNESS), 
                    name=f'Vertical Boundary {i}'
                ))

            fig.update_layout(
                scene=dict(aspectmode="data"), 
                margin=dict(l=0, r=0, b=0, t=40)
            )
            fig.show(renderer=RENDERER_TYPE)  

if __name__ == "__main__":
    streetInstance = Street(RELATIVE_CSV_PATH)
    streetInstance.adjustPoints(SENSOR_PITCH_DEG, SENSOR_YAW_DEG, SENSOR_ROLL_DEG, CAMERA_HEIGHT_OFFSET)
    streetInstance.applyMask(MASK_Z_MIN, MASK_Z_MAX, MASK_Y_MIN, MASK_Y_MAX)
    streetInstance.interpolateCloud(INTERPOLATION_FACTOR)

    streetInstance.findRoad()
    streetInstance.findContours()
    streetInstance.visualize()