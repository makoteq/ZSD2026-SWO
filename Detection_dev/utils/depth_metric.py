import torch
import cv2
import numpy as np
import sys
import os
from typing import Final, Any

ENCODER: Final[str] = 'vits'
FEATURES: Final[int] = 64
OUT_CHANNELS: Final[list[int]] = [48, 96, 192, 384]
MAX_DEPTH: Final[int] = 80
DEVICE_CPU: Final[str] = "cpu"
UINT8_DTYPE: Final[Any] = np.uint8
NORM_MAX: Final[int] = 255
COLORMAP: Final[int] = cv2.COLORMAP_INFERNO


class DepthMetric:
    def __init__(self, modelPath: str, libPath: str) -> None:
        self.device = torch.device(DEVICE_CPU)

        absLibPath = os.path.abspath(os.path.join(libPath, "metric_depth"))
        if absLibPath not in sys.path:
            sys.path.insert(0, absLibPath)

        try:
            from depth_anything_v2.dpt import DepthAnythingV2
        except ImportError:
            raise ImportError(f"Not found depth_anything_v2 in: {absLibPath}")

        self.model = DepthAnythingV2(
            encoder=ENCODER,
            features=FEATURES,
            out_channels=OUT_CHANNELS,
            max_depth=MAX_DEPTH
        )

        absModelPath = os.path.abspath(modelPath)
        if not os.path.exists(absModelPath):
            raise FileNotFoundError(f"Not found weights: {absModelPath}")

        self.model.load_state_dict(torch.load(absModelPath, map_location=DEVICE_CPU))
        self.model.to(self.device)
        self.model.eval()
        print("DepthMetric: model loaded.")

    def getDepthMap(self, frame: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            depth = self.model.infer_image(frame)
        return depth.astype(np.float32)

    def saveDepthMap(self, depthMap: np.ndarray, outputDir: str, name: str = "depth") -> None:
        os.makedirs(outputDir, exist_ok=True)

        npyPath = os.path.join(outputDir, f"{name}.npy")
        np.save(npyPath, depthMap)
        print(f"DepthMetric: saved raw depth map -> {npyPath}")

        depth_norm = (depthMap - depthMap.min()) / (depthMap.max() - depthMap.min())
        depth_vis = (depth_norm * NORM_MAX).astype(UINT8_DTYPE)
        depth_color = cv2.applyColorMap(depth_vis, COLORMAP)
        pngPath = os.path.join(outputDir, f"{name}.png")
        cv2.imwrite(pngPath, depth_color)
        print(f"DepthMetric: saved visualization -> {pngPath}")


def rankCarsByDepth(depthMap: np.ndarray, cars: list[dict]) -> list[dict]:
    """
    cars: list of dicts with keys 'id', 'x1', 'y1', 'x2', 'y2'
    returns: list of dicts with keys 'id', 'depth_m', sorted closest first
    """
    results = []
    h, w = depthMap.shape

    for car in cars:
        x1 = max(0, min(int(car['x1']), w - 1))
        x2 = max(0, min(int(car['x2']), w - 1))
        y1 = max(0, min(int(car['y1']), h - 1))
        y2 = max(0, min(int(car['y2']), h - 1))

        region = depthMap[y1:y2, x1:x2]
        avg_depth_m = float(np.mean(region)) if region.size > 0 else 0.0

        results.append({'id': car['id'], 'depth_m': avg_depth_m})

    results.sort(key=lambda x: x['depth_m']) 
    return results


# CHOOSE 2 of func below for filling the bboxes of detected cars in depth map of first frame,
# but "ALTERNATIVE" is probably better 
def fillBboxesRowMax(depthMap: np.ndarray, bboxes: list[dict], paddingFactor: float = 0.05) -> np.ndarray:
    
    result = depthMap.copy()
    h, w = result.shape

    for bbox in bboxes:
        bboxW = int(bbox['x2']) - int(bbox['x1'])
        bboxH = int(bbox['y2']) - int(bbox['y1'])
        padX = int(bboxW * paddingFactor)
        padY = int(bboxH * paddingFactor)

        x1 = max(0, min(int(bbox['x1']) - padX, w - 1))
        x2 = max(0, min(int(bbox['x2']) + padX, w - 1))
        y1 = max(0, min(int(bbox['y1']) - padY, h - 1))
        y2 = max(0, min(int(bbox['y2']) + padY, h - 1))

        if y2 <= y1 or x2 <= x1:
            continue

        for row in range(y1, y2):
            rowMax = float(np.max(result[row, x1:x2]))
            result[row, x1:x2] = rowMax

    return result


def fillBboxesRowMeanBackground(depthMap: np.ndarray, bboxes: list[dict], paddingFactor: float = 0.05, useMedian: bool = True) -> np.ndarray:
    
    result = depthMap.copy()
    h, w = result.shape

    paddedBboxes = []
    for bbox in bboxes:
        bboxW = int(bbox['x2']) - int(bbox['x1'])
        bboxH = int(bbox['y2']) - int(bbox['y1'])
        padX = int(bboxW * paddingFactor)
        padY = int(bboxH * paddingFactor)

        x1 = max(0, min(int(bbox['x1']) - padX, w - 1))
        x2 = max(0, min(int(bbox['x2']) + padX, w - 1))
        y1 = max(0, min(int(bbox['y1']) - padY, h - 1))
        y2 = max(0, min(int(bbox['y2']) + padY, h - 1))
        paddedBboxes.append((x1, y1, x2, y2))

    for i, bbox in enumerate(bboxes):
        x1, y1, x2, y2 = paddedBboxes[i]

        if y2 <= y1 or x2 <= x1:
            continue

        for row in range(y1, y2):
            bgMask = np.ones(w, dtype=bool)
            for bx1, by1, bx2, by2 in paddedBboxes:
                if by1 <= row < by2:
                    bgMask[bx1:bx2] = False

            bgPixels = depthMap[row, bgMask]

            if bgPixels.size > 0:
                fillValue = float(np.median(bgPixels) if useMedian else np.mean(bgPixels))
            else:
                fillValue = float(np.max(result[row, x1:x2]))

            result[row, x1:x2] = fillValue

    return result




# Alternative: changes depth values for the whole row of the whole depth map based on median from row without pixels from bboxes
def flattenRowsMedianBackground(depthMap: np.ndarray, bboxes: list[dict], paddingFactor: float = 0.05, useMedian: bool = True) -> np.ndarray:
   
    result = depthMap.copy()
    h, w = result.shape

    paddedBboxes = []
    for bbox in bboxes:
        bboxW = int(bbox['x2']) - int(bbox['x1'])
        bboxH = int(bbox['y2']) - int(bbox['y1'])
        padX = int(bboxW * paddingFactor)
        padY = int(bboxH * paddingFactor)

        x1 = max(0, min(int(bbox['x1']) - padX, w - 1))
        x2 = max(0, min(int(bbox['x2']) + padX, w - 1))
        y1 = max(0, min(int(bbox['y1']) - padY, h - 1))
        y2 = max(0, min(int(bbox['y2']) + padY, h - 1))
        paddedBboxes.append((x1, y1, x2, y2))

    for row in range(h):
        bgMask = np.ones(w, dtype=bool)
        for bx1, by1, bx2, by2 in paddedBboxes:
            if by1 <= row < by2:
                bgMask[bx1:bx2] = False

        bgPixels = depthMap[row, bgMask]

        if bgPixels.size == 0:
            continue

        fillValue = float(np.median(bgPixels) if useMedian else np.mean(bgPixels))
        result[row, :] = fillValue  

    return result



def saveDepthVisualization(depthMap: np.ndarray, outputDir: str, name: str = "depth") -> None:
    """Saves only the PNG visualization of a depth map."""
    os.makedirs(outputDir, exist_ok=True)
    depth_norm = (depthMap - depthMap.min()) / (depthMap.max() - depthMap.min())
    depth_vis = (depth_norm * NORM_MAX).astype(UINT8_DTYPE)
    depth_color = cv2.applyColorMap(depth_vis, COLORMAP)
    pngPath = os.path.join(outputDir, f"{name}.png")
    cv2.imwrite(pngPath, depth_color)
    print(f"DepthMetric: saved visualization -> {pngPath}")