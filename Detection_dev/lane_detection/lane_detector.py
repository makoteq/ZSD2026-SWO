import cv2
import numpy as np

class LaneDetector:
    """Utility class for lane detection and classification."""

    def get_roi(self, sample_image):
        """
        Interactive tool to define a 4-point Region of Interest (ROI).
        Returns a polygon array compatible with cv2.fillPoly.
        """
        roi_points = []
        temp_img = sample_image.copy()
        window_name = "Select 4 ROI points"

        def click_event(event, x, y, flags, param):
            if event == cv2.EVENT_LBUTTONDOWN:
                roi_points.append((x, y))
                cv2.circle(temp_img, (x, y), 5, (0, 0, 255), -1)
                if len(roi_points) > 1:
                    cv2.line(temp_img, roi_points[-2], roi_points[-1], (0, 255, 0), 2)
                cv2.imshow(window_name, temp_img)

        cv2.imshow(window_name, temp_img)
        cv2.setMouseCallback(window_name, click_event)

        while len(roi_points) < 4:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cv2.destroyWindow(window_name)
        return np.array([roi_points], np.int32)

    def detect_line_type(self, m, b, edges_image, y_min, y_max):
        """
        Scans pixels along the calculated mathematical line to determine its type.
        Returns 'solid', 'dashed', or 'unknown'.
        """
        white_pixels_count = 0
        total_pixels_count = 0

        # Search window width
        window_size = 15

        for y in range(y_min, y_max):
            try:
                x = int((y - b) / m)

                # Boundary check
                if window_size <= x < edges_image.shape[1] - window_size:
                    if np.any(edges_image[y, x - window_size:x + window_size + 1] == 255):
                        white_pixels_count += 1
                total_pixels_count += 1
            except ZeroDivisionError:
                continue

        if total_pixels_count == 0:
            return "unknown"

        fill_ratio = white_pixels_count / total_pixels_count

        # Threshold for dashed vs solid lines
        return "solid" if fill_ratio > 0.40 else "dashed"

    def get_lines(self, frame, roi_mask_poly):
        """
        Processes the frame and extracts lane equations.
        Returns a list of tuples: (slope, intercept, line_type).
        """
        if roi_mask_poly is None or len(roi_mask_poly[0]) == 0:
            return []

        height, width = frame.shape[:2]
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)

        # Low thresholds to catch edges even in high-brightness areas
        edges = cv2.Canny(blur, 20, 80)

        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, roi_mask_poly, 255)
        masked_edges = cv2.bitwise_and(edges, mask)

        lines = cv2.HoughLinesP(
            masked_edges, 1, np.pi / 180, threshold=40,
            minLineLength=20, maxLineGap=200
        )

        lines_data = []
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if x2 == x1:
                    continue

                m = (y2 - y1) / (x2 - x1)
                if abs(m) < 0.5:
                    continue

                b = y1 - m * x1
                x_bottom = (height - b) / m
                lines_data.append((m, b, x_bottom))

        # Cluster similar lines into single lane markings
        clusters = []
        threshold_x = width * 0.08
        for m, b, x_bottom in lines_data:
            added = False
            for cluster in clusters:
                if abs(cluster['x_bottom'] - x_bottom) < threshold_x:
                    cluster['m'].append(m)
                    cluster['b'].append(b)
                    added = True
                    break
            if not added:
                clusters.append({'x_bottom': x_bottom, 'm': [m], 'b': [b]})

        y_max = height
        y_min = np.min(roi_mask_poly[0][:, 1])
        results = []

        for cluster in clusters:
            avg_m = np.mean(cluster['m'])
            avg_b = np.mean(cluster['b'])
            line_type = self.detect_line_type(avg_m, avg_b, masked_edges, y_min, y_max)
            results.append((avg_m, avg_b, line_type))

        return results

    def draw_lanes(self, frame, line_params, roi_poly):
        """
        Draws detected lines onto the frame.
        Solid = Green, Dashed = Red.
        """
        line_image = np.zeros_like(frame)
        height = frame.shape[0]

        y1_draw = height
        y2_draw = 0

        for m, b, line_type in line_params:
            try:
                x1 = int((y1_draw - b) / m)
                x2 = int((y2_draw - b) / m)

                color = (0, 255, 0) if line_type == "solid" else (0, 0, 255)
                cv2.line(line_image, (x1, y1_draw), (x2, y2_draw), color, 5)
            except ZeroDivisionError:
                continue

        combined = cv2.addWeighted(frame, 1.0, line_image, 0.8, 0)
        cv2.polylines(combined, [roi_poly], True, (255, 0, 0), 2)

        return combined