from pathlib import Path

from algorithms.lane_detection_brute.lane_detection_brute import runLaneDetection
from utils.points import build_lines_equations


if __name__ == "__main__":
		project_root = Path(__file__).resolve().parents[1]
		cached_lanes_path = project_root / "data" / "output" / "lines.json"
		print(f"cached_lanes_path = {cached_lanes_path}")
		if cached_lanes_path.exists():
			lines_path = cached_lanes_path
		elif cached_lanes_path.exists():
			lines_path = cached_lanes_path
		else:
			lines_path = runLaneDetection(showVideo=False,PASSES_COUNT=5)

		detected_lines = build_lines_equations(lines_path)
		print(f"detected_lines = {detected_lines}")
