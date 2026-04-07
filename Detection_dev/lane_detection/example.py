import os
import cv2

from lane_detector import LaneDetector

def main():
    """Execution example."""
    input_file = "data_input/img_1.png"
    out_dir = "results"

    if not os.path.exists(out_dir):
        os.makedirs(out_dir)

    img = cv2.imread(input_file)
    if img is None:
        print(f"Image not found: {input_file}")
        return

    detector = LaneDetector()

    # 1. Define ROI manually
    roi_poly = detector.get_roi(img)

    # 2. Extract mathematical representations of lanes
    lines_equations = detector.get_lines(img, roi_poly)

    # 3. Render results
    result_img = detector.draw_lanes(img, lines_equations, roi_poly)

    # 4. Save output
    save_path = os.path.join(out_dir, os.path.basename(input_file))
    cv2.imwrite(save_path, result_img)
    print(f"[INFO] Saved result to: {save_path}")


if __name__ == "__main__":
    main()