import cv2
from pathlib import Path
from lane_detector import LaneDetector

def load_images(path: str):
    images = []
    for file_path in Path(path).iterdir():
        img = cv2.imread(str(file_path))
        if img is not None:
            images.append(img)

    return images

def show_image_with_trackbar(window_name, image):
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    def trackbar_change(pos):
        if pos < 1:
            pos = 1

        scale = pos / 100.0
        h = int(image.shape[0] * scale)
        w = int(image.shape[1] * scale)

        rsz_image = cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)

        cv2.resizeWindow(window_name, w, h)
        cv2.imshow(window_name, rsz_image)

    cv2.createTrackbar('Scale %', window_name, 25, 200, trackbar_change)

    trackbar_change(25)  # Start with 25% scale

    cv2.waitKey(0)
    cv2.destroyWindow(window_name)

def main():
    detector = LaneDetector()
    images = load_images("frames")

    for img in images:
        lanes = detector.detect(img)
        res = detector.draw_on_original(img, lanes)
        show_image_with_trackbar("result", res)

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()