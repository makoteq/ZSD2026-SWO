import torch
import cv2
import numpy as np
import sys
import os
from typing import Final, Any
import matplotlib.pyplot as plt

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
            depthData = depth.astype(np.float32)

            plt.imshow(depthData, cmap='magma', interpolation='nearest')
            plt.colorbar(label='Depth')
            plt.title('Depth Map Heatmap')
            plt.axis('off')
            plt.show()

        return depthData

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

# 3 of functions below are responsible for the same thing: filling the bboxes in the depth map when car is detecte to make clear depth map for the first frame
# Choose one of them

def fillBboxesRowMin(depthMap: np.ndarray, bboxes: list[dict], paddingFactor: float = 0.05) -> np.ndarray:
    """
    For each bbox, fills the area of the bbox row by row, with the minimum value of each row in that bbox, 
    """
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
            rowMin = float(np.min(result[row, x1:x2]))
            result[row, x1:x2] = rowMin

    return result

def fillBboxesRowMinMasked(depthMap: np.ndarray, bboxes: list[dict], paddingFactor: float = 0.05, maskDilation: int = 15) -> np.ndarray:
    """
    Get the contour of the car inside the bbox and fill only the interior of that contour with the whole bbox row minimum, 
    Outiside the contour, inside the bbox remains with the original depth values.
    """
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

        region = result[y1:y2, x1:x2]
        if region.size == 0:
            continue

        roi_min, roi_max = region.min(), region.max()
        if roi_max == roi_min:
            continue
            
        roi_norm = ((region - roi_min) / (roi_max - roi_min) * 255).astype(np.uint8)
        _, binary_mask = cv2.threshold(roi_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            car_mask = np.zeros_like(binary_mask)
            cv2.drawContours(car_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
            if maskDilation > 0:
                kernel = np.ones((maskDilation, maskDilation), np.uint8)
                car_mask = cv2.dilate(car_mask, kernel, iterations=1)

            # Each row inside the contour gets the min value of that row in the original region (bbox area)
            rowMins = np.min(region, axis=1, keepdims=True)
            artificial_bg = np.broadcast_to(rowMins, region.shape)
            
            result[y1:y2, x1:x2] = np.where(car_mask == 255, artificial_bg, region)

    return result

def fillBboxesRowMinMasked_hybrid(depthMap: np.ndarray, bboxes: list[dict], paddingFactor: float = 0.05, maskDilation: int = 15) -> np.ndarray:
    """
    Get the contour of the car inside the bbox and fill only the interior of that contour 
    with mean value from the area between the contour and the bbox edge, if there are at least minBgPixels pixels of background,
    otherwise fill with the whole bbox row minimum, 
    Outiside the contour, inside the bbox remains with the original depth values.
    """
    minBgPixels = 5

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

        region = result[y1:y2, x1:x2]
        if region.size == 0:
            continue

        roi_min, roi_max = region.min(), region.max()
        if roi_max == roi_min:
            continue
            
        roi_norm = ((region - roi_min) / (roi_max - roi_min) * 255).astype(np.uint8)
        _, binary_mask = cv2.threshold(roi_norm, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            car_mask = np.zeros_like(binary_mask)
            cv2.drawContours(car_mask, [largest_contour], -1, 255, thickness=cv2.FILLED)
            if maskDilation > 0:
                kernel = np.ones((maskDilation, maskDilation), np.uint8)
                car_mask = cv2.dilate(car_mask, kernel, iterations=1)

            # Each row inside the contour gets either the mean value from the area between the contour and the bbox edge (if enough background pixels) or the min value of that row in the original region (bbox area)            
            only_background = region.copy()
            only_background[car_mask == 255] = np.nan
            bg_pixels_count = np.sum(~np.isnan(only_background), axis=1, keepdims=True)

            with np.errstate(all='ignore'):
                row_means = np.nanmean(only_background, axis=1, keepdims=True)
            
            row_mins = np.min(region, axis=1, keepdims=True)
            hybrid_rows = np.where(bg_pixels_count >= minBgPixels, row_means, row_mins)
            hybrid_rows = np.nan_to_num(hybrid_rows, nan=np.nanmin(row_mins))

            artificial_bg = np.broadcast_to(hybrid_rows, region.shape)
            result[y1:y2, x1:x2] = np.where(car_mask == 255, artificial_bg, region)

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
    print(f"DepthV2: saved visualization -> {pngPath}")