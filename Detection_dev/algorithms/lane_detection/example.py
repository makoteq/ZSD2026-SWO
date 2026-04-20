import cv2
import os
from lane_detector import LaneDetector


def main():
    detector = LaneDetector('ROI_v2.pt')

    image_paths = [
        "test_images/1.png", "test_images/2.png", "test_images/3.png",
        "test_images/4.png", "test_images/5.webp", "test_images/7.webp",
        "test_images/8.jpg", "test_images/9.webp", "test_images/10.jpg",
        "test_images/11.jpg", "test_images/12.jpg", "test_images/14.png",
    ]

    for img_path in image_paths:
        if not os.path.exists(img_path):
            print(f"Not found: {img_path}")
            continue

        print(f"\nAnalyzing: {img_path}")

        # Load img
        img = cv2.imread(img_path)

        # Extract mathematical parameters
        detected_lines = detector.process_image(img, debug=True)

        # Log coefficients
        if detected_lines:
            print(f"Lanes found: {len(detected_lines)}")
            for i, line in enumerate(detected_lines):
                print(f" L{i + 1}: x = {line['m']:.4f} * y + {line['b']:.2f}")
        else:
            print("No lanes found.")

        # Draw infinite mathematical lines
        final_output = detector.draw_lanes(img, detected_lines)

        # Scale down the output window for better display
        target_width = 800
        aspect_ratio = target_width / final_output.shape[1]
        target_height = int(final_output.shape[0] * aspect_ratio)
        resized_output = cv2.resize(final_output, (target_width, target_height))

        # Show final overlay
        cv2.imshow("Final Result", resized_output)

        # Press 'q' to quit, any other key for the next image
        if cv2.waitKey(0) & 0xFF == ord('q'):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()