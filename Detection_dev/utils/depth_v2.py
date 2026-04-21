import torch
import cv2
import numpy as np
import sys
import os
from typing import Final, Any

ENCODER: Final[str] = 'vits'
FEATURES: Final[int] = 64
OUT_CHANNELS: Final[list[int]] = [48, 96, 192, 384]
DEVICE_CPU: Final[str] = "cpu"
UINT8_DTYPE: Final[Any] = np.uint8
NORM_MAX: Final[int] = 255
COLORMAP: Final[int] = cv2.COLORMAP_INFERNO

class DepthV2:
    def __init__(self, modelPath: str, libPath: str) -> None:
        self.device = torch.device(DEVICE_CPU)

        absLibPath = os.path.abspath(libPath)
        if absLibPath not in sys.path:
            sys.path.insert(0, absLibPath)

        try:
            from depth_anything_v2.dpt import DepthAnythingV2
        except ImportError:
            raise ImportError(f"Not found depth_anything_v2 in: {absLibPath}")

        self.model = DepthAnythingV2(
            encoder=ENCODER,
            features=FEATURES,
            out_channels=OUT_CHANNELS
        )

        absModelPath = os.path.abspath(modelPath)
        if not os.path.exists(absModelPath):
            raise FileNotFoundError(f"Not found weights: {absModelPath}")

        self.model.load_state_dict(torch.load(absModelPath, map_location=DEVICE_CPU))
        self.model.to(self.device)
        self.model.eval()
        print("DepthV2: model loaded.")

    def getDepthMap(self, frame: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            depth = self.model.infer_image(frame)
        return depth.astype(np.float32)

    def saveDepthMap(self, depthMap: np.ndarray, outputDir: str, name: str = "depth") -> None:
        os.makedirs(outputDir, exist_ok=True)

        npyPath = os.path.join(outputDir, f"{name}.npy")
        np.save(npyPath, depthMap)
        print(f"DepthV2: saved raw depth map -> {npyPath}")

        depth_norm = (depthMap - depthMap.min()) / (depthMap.max() - depthMap.min())
        depth_vis = (depth_norm * NORM_MAX).astype(UINT8_DTYPE)
        depth_color = cv2.applyColorMap(depth_vis, COLORMAP)
        pngPath = os.path.join(outputDir, f"{name}.png")
        cv2.imwrite(pngPath, depth_color)
        print(f"DepthV2: saved visualization -> {pngPath}")

def rankCarsByDepth(depthMap: np.ndarray, cars: list[dict]) -> list[dict]:
    """
    cars: list of dicts with keys 'id', 'x1', 'y1', 'x2', 'y2'
    returns: list of dicts with keys 'id', 'depth', sorted by depth (closest first)
    """
    results = []
    h, w = depthMap.shape

    for car in cars:
        x1 = max(0, min(int(car['x1']), w - 1))
        x2 = max(0, min(int(car['x2']), w - 1))
        y1 = max(0, min(int(car['y1']), h - 1))
        y2 = max(0, min(int(car['y2']), h - 1))

        region = depthMap[y1:y2, x1:x2]
        avg_depth = float(np.mean(region)) if region.size > 0 else 0.0

        results.append({'id': car['id'], 'depth': avg_depth})

    results.sort(key=lambda x: x['depth'], reverse=True)
    return results

def rankCarsObjectsByDepth(depthMap: np.ndarray, cars: list[dict], carsDict: dict) -> list:
    """
    cars: list of dicts with keys 'id', 'x1', 'y1', 'x2', 'y2' (from current frame YOLO)
    carsDict: dictionary of Car objects
    returns: sorted list of Car objects from closest to farthest
    """
    results = []
    h, w = depthMap.shape

    for car in cars:
        if car['id'] not in carsDict:
            continue

        x1 = max(0, min(int(car['x1']), w - 1))
        x2 = max(0, min(int(car['x2']), w - 1))
        y1 = max(0, min(int(car['y1']), h - 1))
        y2 = max(0, min(int(car['y2']), h - 1))

        region = depthMap[y1:y2, x1:x2]
        avg_depth = float(np.mean(region)) if region.size > 0 else 0.0

        results.append((carsDict[car['id']], avg_depth))

    results.sort(key=lambda x: x[1], reverse=True)
    return [car for car, _ in results]