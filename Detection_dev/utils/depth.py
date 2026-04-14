import torch
import cv2
import numpy as np
import time
import psutil
import os
import sys
from typing import Any, Final

ENCODER: Final[str] = 'vits'
FEATURES: Final[int] = 64
OUT_CHANNELS: Final[list[int]] = [48, 96, 192, 384]
DEVICE_CPU: Final[str] = "cpu"
UINT8_DTYPE: Final[Any] = np.uint8
NORM_MAX: Final[int] = 255

class Depth:
    def __init__(self, modelPath: str, libPath: str) -> None:
        self.device: torch.device = torch.device(DEVICE_CPU)
        self.totalProcessingTime: float = 0.0
        self.totalFramesProcessed: int = 0
        self.cpuUsageSamples: list[float] = []

        absLibPath: str = os.path.abspath(libPath)
        
        if absLibPath not in sys.path:
            sys.path.insert(0, absLibPath)

        # Wymuszenie ścieżki do wewnętrznego folderu modułu
        innerModulePath: str = os.path.join(absLibPath, "depth_anything_v2")
        if os.path.exists(innerModulePath) and innerModulePath not in sys.path:
            sys.path.insert(0, innerModulePath)

        try:
            from depth_anything_v2.dpt import DepthAnythingV2
        except ImportError:
            from dpt import DepthAnythingV2

        self.model = DepthAnythingV2(
            encoder=ENCODER,
            features=FEATURES,
            out_channels=OUT_CHANNELS
        )

        absModelPath: str = os.path.abspath(modelPath)
        if not os.path.exists(absModelPath):
            raise FileNotFoundError(f"Weight file not found: {absModelPath}")

        self.model.load_state_dict(torch.load(absModelPath, map_location=DEVICE_CPU))
        self.model.to(self.device)
        self.model.eval()

    def getDepthMap(self, frame: np.ndarray) -> np.ndarray:
        startTime: float = time.perf_counter()
        cpuBefore: float = psutil.cpu_percent(interval=None)

        with torch.no_grad():
            depth = self.model.infer_image(frame)

        depthOutput: np.ndarray = depth.astype(np.float32)

        endTime: float = time.perf_counter()
        cpuAfter: float = psutil.cpu_percent(interval=None)

        self.totalProcessingTime += (endTime - startTime)
        self.totalFramesProcessed += 1
        self.cpuUsageSamples.append((cpuBefore + cpuAfter) / 2)

        return depthOutput

    def getLog(self) -> None:
        avgCpu: float = sum(self.cpuUsageSamples) / len(self.cpuUsageSamples) if self.cpuUsageSamples else 0.0
        print(f"--- Depth Anything V2 Log ---")
        print(f"Frames: {self.totalFramesProcessed}")
        print(f"Total Time: {self.totalProcessingTime:.4f}s")
        print(f"Avg Time/Frame: {(self.totalProcessingTime / self.totalFramesProcessed if self.totalFramesProcessed > 0 else 0):.4f}s")
        print(f"Avg CPU: {avgCpu:.2f}%")
        print(f"-----------------------------")

    def visualizeDepth(self, depthMap: np.ndarray) -> np.ndarray:
        minVal: float = depthMap.min()
        maxVal: float = depthMap.max()
        if maxVal - minVal == 0:
            return np.zeros(depthMap.shape, dtype=UINT8_DTYPE)
        depthNormalized: np.ndarray = (depthMap - minVal) / (maxVal - minVal)
        return (depthNormalized * NORM_MAX).astype(UINT8_DTYPE)