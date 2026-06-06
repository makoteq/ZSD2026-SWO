import torch
import cv2
import numpy as np
import sys
import os
from typing import Final, Any

DEPTH_DEBUG: bool = True


ENCODER: Final[str] = 'vits'
FEATURES: Final[int] = 64
OUT_CHANNELS: Final[list[int]] = [48, 96, 192, 384]
DEVICE_CPU: Final[str] = "cpu"
UINT8_DTYPE: Final[Any] = np.uint8
NORM_MAX: Final[int] = 255
COLORMAP: Final[int] = cv2.COLORMAP_INFERNO

class DepthV2:
    def __init__(self, modelPath: str, libPath: str) -> None:
        """ Loads and initializes the DepthAnythingV2 model.

        Args:
            modelPath (str): Path to the model weights file
            libPath (str): Path to the DepthAnythingV2 library directory

        Raises:
            ImportError: If the DepthAnythingV2 library is not found in libPath
            FileNotFoundError: If the model weights file does not exist
        """
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
        """ Runs inference on a single frame and returns the raw depth map.

        Args:
            frame (np.ndarray): Input image as a BGR array of shape (H, W, 3)
        Returns:
            np.ndarray: Raw depth map as a float32 array of shape (H, W)
        """
        with torch.no_grad():
            depth = self.model.infer_image(frame)
            depthData = depth.astype(np.float32)

        return depthData


def computeDepthMap(npyPath: str, firstFrame: np.ndarray, modelPath: str, libPath: str, outputDir: str) -> np.ndarray:
    """ 
    Runs depth estimation inference on a single frame and returns the raw depth map. Saves a debug PNG if DEPTH_DEBUG is enabled.

    Args:
        npyPath (str): Path to the .npy file used to check if depth map already exists
        firstFrame (np.ndarray): First frame of the video on which inference is performed
        modelPath (str): Path to the DepthAnythingV2 model weights file
        libPath (str): Path to the DepthAnythingV2 library directory
        outputDir (str): Directory where the debug PNG is saved if DEPTH_DEBUG is enabled

    Returns:
        np.ndarray: Raw depth map (where depth images of possible vehicles can be visible) as a float32 array.
    """
    if not os.path.exists(npyPath):
        print("DepthV2: computing depth map from scratch.")
        depthProcessor = DepthV2(modelPath=modelPath, libPath=libPath)
        rawDepthMap = depthProcessor.getDepthMap(firstFrame)
        if DEPTH_DEBUG:
            saveDepthDebugPng(rawDepthMap, outputDir, name="raw_depth")
        return rawDepthMap
    else:
        print("DepthV2: depth map already exists, computing should be done via setup script.")


def removeVehiclesFromDepthMap(depthMap: np.ndarray, bboxes: list[dict], paddingFactor: float = 0.05, useMedian: bool = True) -> np.ndarray:
    """ 
    Cleans a raw depth map of cars depth images by replacing every row with a value equal to the median or mean of that row's background pixels, excluding detected bounding boxes.

    Args:
        depthMap (np.ndarray): Raw depth map where depth images of possible vehicles can be visible
        bboxes (list[dict]): List of bounding boxes of possible vehicles
        paddingFactor (float): Fraction of bounding box size as padding
        useMedian (bool): Median or mean value

    Returns:
        np.ndarray: Depth map where every row is filled with a uniform value equal to the median or mean of that row's background pixels.
    """
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


def saveDepthMap(depthMap: np.ndarray, npyDir: str, outputDir: str, name: str = "base_depth") -> None:
    """ 
    Saves the depth map to a .npy file and optionally saves a debug PNG if DEPTH_DEBUG is enabled.

    Args:
        depthMap (np.ndarray): Depth map to save
        npyDir (str): Where the .npy file is saved
        outputDir (str): Where the debug PNG is saved if DEPTH_DEBUG is enabled
    """
    os.makedirs(npyDir, exist_ok=True)
    npyPath = os.path.join(npyDir, f"{name}.npy")
    np.save(npyPath, depthMap)

    if DEPTH_DEBUG:
        print(f"DepthV2: saved depth map -> {npyPath}")
        saveDepthDebugPng(depthMap, outputDir, name=name)


def loadDepthMap(npyPath: str) -> np.ndarray:
    """ Loads a precomputed depth map from a .npy file.
    Args:
        npyPath (str): Path to the .npy file containing the depth map
    Returns:
        np.ndarray: Depth map as a float32 array of shape (H, W)
    """
    if not os.path.exists(npyPath):
        raise FileNotFoundError(f"Depth map not found: {npyPath}. Run setup.py first.")
    print(f"DepthV2: loaded depth map from {npyPath} file.")
    return np.load(npyPath)


def rankCarsByDepth(depthMap: np.ndarray, cars: list[dict]) -> list[dict]:
    """ Ranks vehicles by their average depth value extracted from the depth map.

    Args:
        depthMap (np.ndarray): Depth map as a float32 array of shape (H, W)
        cars (list[dict]): List of detected vehicles, each with keys 'id', 'x1', 'y1', 'x2', 'y2'.

    Returns:
        list[dict]: List of dicts with keys 'id' and 'depth', sorted by depth.
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


def saveDepthDebugPng(depthMap: np.ndarray, outputDir: str, name: str = "depth") -> None:
    """ Saves a colorized PNG visualization of a depth map for debugging purposes.
    Args:
        depthMap (np.ndarray): Depth map to visualize
        outputDir (str): Directory where the PNG file is saved
    """
    os.makedirs(outputDir, exist_ok=True)
    depth_norm = (depthMap - depthMap.min()) / (depthMap.max() - depthMap.min())
    depth_vis = (depth_norm * NORM_MAX).astype(UINT8_DTYPE)
    depth_color = cv2.applyColorMap(depth_vis, COLORMAP)
    pngPath = os.path.join(outputDir, f"{name}.png")
    cv2.imwrite(pngPath, depth_color)
    print(f"DepthV2: saved debug PNG -> {pngPath}")