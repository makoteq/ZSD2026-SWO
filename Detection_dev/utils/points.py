import json
from pathlib import Path
from typing import Any


def _line_equation_from_points(start: list[int], end: list[int]) -> dict[str, Any]:
	
	# - canonical form: A*x + B*y + C = 0
	# - slope form: y = m*x + b

	x1, y1 = start
	x2, y2 = end

	a = y2 - y1
	b = x1 - x2
	c = x2 * y1 - x1 * y2

	equation: dict[str, Any] = {
		"A": a,
		"B": b,
		"C": c,
		"standard_form": f"{a}*x + {b}*y + {c} = 0",
	}

	if x1 == x2:
		equation["x"] = x1
		equation["slope_intercept"] = None
	else:
		m = (y2 - y1) / (x2 - x1)
		intercept = y1 - m * x1
		equation["m"] = m
		equation["b"] = intercept
		equation["slope_intercept"] = f"y = {m}*x + {intercept}"

	return equation


def build_lines_equations(lines_json: str | Path | dict[str, Any]) -> dict[str, Any]:

	if isinstance(lines_json, (str, Path)):
		with open(lines_json, "r", encoding="utf-8") as file:
			data = json.load(file)
	else:
		data = lines_json

	result: dict[str, Any] = {
		"passes_threshold": data.get("passes_threshold"),
		"equations": {},
	}

	for line_name in ("left_line", "right_line", "middle_line"):
		line = data[line_name]
		result["equations"][line_name] = _line_equation_from_points(
			line["start"],
			line["end"],
		)

	return result

