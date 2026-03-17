import cv2
import numpy as np
import os
import glob

def process_frame(frame):
    height, width = frame.shape[:2]
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)

    mask = np.zeros_like(edges)
    polygon = np.array([[
        (int(width * 0.30), height),
        (int(width * 0.58), height),
        (int(width * 0.52), int(height * 0.35)),
        (int(width * 0.40), int(height * 0.35))
    ]], np.int32)
    cv2.fillPoly(mask, polygon, 255)
    masked_edges = cv2.bitwise_and(edges, mask)

    lines = cv2.HoughLinesP(
        masked_edges,
        rho=1,
        theta=np.pi / 180,
        threshold=40,
        minLineLength=20,
        maxLineGap=200
    )

    line_image = np.zeros_like(frame)
    if lines is not None:
        lines_data = []
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 == x1: continue
            m = (y2 - y1) / (x2 - x1)
            if abs(m) < 0.8: continue
            b = y1 - m * x1
            x_bottom = (height - b) / m
            lines_data.append((m, b, x_bottom))

        clusters = []
        threshold_x = width * 0.06
        for m, b, x_bottom in lines_data:
            added = False
            for cluster in clusters:
                if abs(cluster['x_bottom'] - x_bottom) < threshold_x:
                    cluster['m'].append(m)
                    cluster['b'].append(b)
                    cluster['x_bottom'] = np.mean([(height - cb) / cm for cm, cb in zip(cluster['m'], cluster['b'])])
                    added = True
                    break
            if not added:
                clusters.append({'x_bottom': x_bottom, 'm': [m], 'b': [b]})

        y1_draw, y2_draw = height, int(height * 0.35)
        for cluster in clusters:
            avg_m, avg_b = np.mean(cluster['m']), np.mean(cluster['b'])
            x1_draw = int((y1_draw - avg_b) / avg_m)
            x2_draw = int((y2_draw - avg_b) / avg_m)
            cv2.line(line_image, (x1_draw, y1_draw), (x2_draw, y2_draw), (0, 255, 0), 5)

    return cv2.addWeighted(frame, 1.0, line_image, 1.0, 0)

def main():
    DIR_IN = "data_input"
    DIR_OUT = "results"

    if not os.path.exists(DIR_OUT):
        os.makedirs(DIR_OUT)

    files = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        files.extend(glob.glob(os.path.join(DIR_IN, ext)))

    if not files:
        print(f"No images found in {DIR_IN}")
        return

    print(f"Processing {len(files)} images with OpenCV...")
    for f_path in files:
        frame = cv2.imread(f_path)
        if frame is None: continue
        processed = process_frame(frame)
        cv2.imwrite(os.path.join(DIR_OUT, os.path.basename(f_path)), processed)

if __name__ == "__main__":
    main()