import cv2
import os
from lane_detector import LaneDetector


def main():
    # Initialize the detector
    detector = LaneDetector()

    # Path to a single image file
    img_path = "data_input/1.png"

    if not os.path.exists(img_path):
        print(f"[ERROR] File not found: {img_path}")
        return

    print(f"--- Processing: {img_path} ---")

    # 1. Load the image
    img = cv2.imread(img_path)

    # 2. Detect lines
    detected_lines = detector.detect(img)

    # 3. Print results to the console
    if not detected_lines:
        print("No lines detected with >= 70% confidence.")
    else:
        for i, line in enumerate(detected_lines):
            print(f"Line {i + 1}: y = {line['a']:.2f}x + {line['b']:.2f} "
                  f"| Type: {line['type'].upper()} "
                  f"| Confidence: {line['score'] * 100:.1f}%")

    # 4. Draw lines on the image (Green = solid, Red = dashed)
    result_img = detector.draw_lines(img, detected_lines)

    # Scale the window down if the image is too large
    h, w = result_img.shape[:2]
    if h > 800:
        result_img = cv2.resize(result_img, (int(w * (800 / h)), 800))

    # Display the result
    cv2.imshow("Lane Detection Result", result_img)

    print("Press any key to close the window and exit.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()