import json

from algorithms.lane_detection_brute.lane_detection_brute import runLaneDetection
from utils.points import build_lines_equations


if __name__ == "__main__":
		lines_path = runLaneDetection(showVideo=False,PASSES_COUNT=5)
		result = build_lines_equations(lines_path)
		print(json.dumps(result, indent=2, ensure_ascii=False))

