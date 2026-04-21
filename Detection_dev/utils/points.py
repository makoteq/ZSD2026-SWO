import json
from pathlib import Path
from typing import Any


def _line_params_from_points(start: list[int], end: list[int]) -> tuple[float, float] | None:
	x1, y1 = start
	x2, y2 = end

	if y1 == y2:
		return None

	m = (x2 - x1) / (y2 - y1)
	intercept = x1 - m * y1
	return float(m), float(intercept)


def _x_bottom_from_points(start: list[int], end: list[int]) -> float:
	return float(start[0] if start[1] >= end[1] else end[0])


def build_lines_equations(
	lines_json: str | Path | dict[str, Any],
) -> list[dict[str, float]]:

	if isinstance(lines_json, (str, Path)):
		with open(lines_json, "r", encoding="utf-8") as file:
			data = json.load(file)
	else:
		data = lines_json

	detected_lines: list[dict[str, float]] = []

	for line_name in ("left_line", "right_line", "middle_line"):
		line = data.get(line_name)
		if not line:
			continue

		line_params = _line_params_from_points(line["start"], line["end"])
		if line_params is None:
			continue

		m, intercept = line_params

		x_bot = _x_bottom_from_points(line["start"], line["end"])

		detected_lines.append(
			{
				"name": line_name,
				"m": m,
				"b": intercept,
				"x_bot": x_bot,
				"abs_m": abs(m),
			}
		)

	return detected_lines

