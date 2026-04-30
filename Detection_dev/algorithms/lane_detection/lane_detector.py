import cv2
import numpy as np

class LaneDetector:
    def __init__(self, config=None):
        self.config = {
            'yolo_conf': 0.25, # Minimal confidence for YOLO detections
            'crop_top': 0.3,  # Percentage of the image height to crop from the top (to focus on the road)
            'adapt_block_size': 81, # Must be odd and >1 for adaptive thresholding
            'adapt_c': -15, # Constant subtracted from the mean or weighted mean in adaptive thresholding
            'hough_threshold_percentage': 0.09, # of the image diagonal,
            'hough_min_length_percentage': 0.2, # of the image diagonal,
            'hough_max_gap_percentage': 0.05, # of the image diagonal,
            'hough_threshold': 0,  # To be calculated based on image diagonal
            'hough_min_length': 0, # To be calculated based on image diagonal
            'hough_max_gap': 0,    # To be calculated based on image diagonal
            'vp_tolerance_percentage': 0.2, # of the image diagonal, for vanishing point consistency check
            'vp_tolerance': 0 # To be calculated based on image diagonal
        }
        if config:
            self.config.update(config)

    def scale_params(self, image):
        height, width = image.shape[:2]
        diagonal = np.sqrt(width**2 + height**2)

        self.config['hough_threshold'] = int(self.config['hough_threshold_percentage'] * diagonal)
        self.config['hough_min_length'] = int(self.config['hough_min_length_percentage'] * diagonal)
        self.config['hough_max_gap'] = int(self.config['hough_max_gap_percentage'] * diagonal)
        self.config['vp_tolerance'] = int(self.config['vp_tolerance_percentage'] * diagonal)

    def crop_image(self, image):
        h = image.shape[0]
        crop_h = int(h * self.config['crop_top'])
        res = image[crop_h:, :]
        return res


    def get_adaptive_mask(self, img):
        height, width = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lower_yellow = np.array([15, 40, 100])
        upper_yellow = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        lower_white = np.array([0, 0, 245])
        upper_white = np.array([180, 20, 255])
        pure_white_mask = cv2.inRange(hsv, lower_white, upper_white)

        mean_brightness = np.mean(gray)

        if mean_brightness > 200:
            p98 = np.percentile(gray, 98)
            thresh_val = p98 - 6
            _, paint_mask = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)

            kernel_width = int(width * 0.15)
            horizontal_kernel = np.ones((1, kernel_width), np.uint8)

            glare_blobs = cv2.morphologyEx(paint_mask, cv2.MORPH_OPEN, horizontal_kernel)
            lines_only = cv2.bitwise_xor(paint_mask, glare_blobs)

            white_mask = cv2.morphologyEx(lines_only, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        else:
            blur = cv2.GaussianBlur(gray, (11, 11), 0)
            white_mask = cv2.adaptiveThreshold(
                blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                self.config['adapt_block_size'],
                self.config['adapt_c']
            )

        combined_paint_mask = cv2.bitwise_or(white_mask, yellow_mask)
        raw_mask = cv2.bitwise_or(combined_paint_mask, pure_white_mask)

        stencil_mask = np.zeros_like(raw_mask)

        lines = cv2.HoughLinesP(
            raw_mask,
            1,
            np.pi / 180,
            threshold=self.config['hough_threshold'],
            minLineLength=self.config['hough_min_length'],
            maxLineGap=self.config['hough_max_gap']
        )

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]

                dy = abs(y2 - y1)
                dx = abs(x2 - x1)

                if dy < dx * 0.5:
                    continue


                cv2.line(stencil_mask, (x1, y1), (x2, y2), 255, 5)


        original_pixels_kept = cv2.bitwise_and(raw_mask, stencil_mask)


        kernel_dilate = np.ones((5, 5), np.uint8)
        final_mask = cv2.dilate(original_pixels_kept, kernel_dilate, iterations=1)

        return final_mask

    def get_math_lines(self, paint_mask):
        height, width = paint_mask.shape

        lines = cv2.HoughLinesP(
            paint_mask,
            rho=1,
            theta=np.pi / 180,
            threshold=self.config['hough_threshold'],
            minLineLength=self.config['hough_min_length'],
            maxLineGap=self.config['hough_max_gap']
        )

        if lines is None:
            return []

        candidates = []
        for line in lines:
            x1, y1, x2, y2 = line[0]

            if abs(y2 - y1) < 10:
                continue

            m_inv = (x2 - x1) / (y2 - y1)
            b_inv = x1 - m_inv * y1

            if abs(m_inv) > 2.0:
                continue

            length = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
            x_bottom = m_inv * height + b_inv

            pts = []
            steps = max(1, int(length / 5))
            for i in range(steps + 1):
                t = i / steps
                px = int(x1 + t * (x2 - x1))
                py = int(y1 + t * (y2 - y1))
                pts.append((px, py))

            candidates.append({
                'm': m_inv, 'b': b_inv, 'weight': length, 'x_bot': x_bottom,
                'pts': pts
            })

        unique_lines = []
        y_eval = height * 0.7

        for cand in sorted(candidates, key=lambda x: x['weight'], reverse=True):
            is_duplicate = False
            cand_x = cand['m'] * y_eval + cand['b']

            for ul in unique_lines:
                ul_x = ul['m'] * y_eval + ul['b']
                if abs(cand_x - ul_x) < width * 0.04 and abs(cand['m'] - ul['m']) < 0.15:
                    is_duplicate = True
                    ul['weight'] += cand['weight']
                    ul['pts'].extend(cand['pts'])
                    break

            if not is_duplicate:
                unique_lines.append(cand)

        if len(unique_lines) < 2:
            return unique_lines

        for ul in unique_lines:
            pts_array = np.array(ul['pts'], dtype=np.float32).reshape((-1, 1, 2))

            [vx, vy, x0, y0] = cv2.fitLine(pts_array, cv2.DIST_HUBER, 0, 0.01, 0.01)

            if vy[0] != 0:
                new_m = vx[0] / vy[0]
                new_b = x0[0] - new_m * y0[0]

                ul['m'] = new_m
                ul['b'] = new_b
                ul['x_bot'] = new_m * height + new_b

            y_coords = pts_array[:, 0, 1]
            ul['weight'] = float(np.max(y_coords) - np.min(y_coords))

        unique_lines.sort(key=lambda x: x['weight'], reverse=True)

        import itertools
        best_combo = None
        best_combo_score = 0
        best_fallback_duo = [unique_lines[0], unique_lines[1]]

        if len(unique_lines) >= 3:
            for combo in itertools.combinations(unique_lines, 3):
                combo_sorted = sorted(combo, key=lambda x: x['weight'], reverse=True)
                l1, l2, l3 = combo_sorted[0], combo_sorted[1], combo_sorted[2]

                dm = l1['m'] - l2['m']

                if abs(dm) < 0.001:
                    continue

                vp_y = (l2['b'] - l1['b']) / dm
                vp_x = l1['m'] * vp_y + l1['b']

                if vp_y > height * 0.8:
                    continue

                l3_x_at_vp = l3['m'] * vp_y + l3['b']
                distance_to_vp = abs(l3_x_at_vp - vp_x)

                if distance_to_vp <= self.config['vp_tolerance']:
                    score = l1['weight'] + l2['weight'] + l3['weight']

                    if score > best_combo_score:
                        best_combo_score = score
                        dy = vp_y - height
                        if dy != 0:
                            new_m = (vp_x - l3['x_bot']) / dy
                            new_b = l3['x_bot'] - new_m * height

                            l1_c, l2_c, l3_c = dict(l1), dict(l2), dict(l3)
                            l3_c['m'] = new_m
                            l3_c['b'] = new_b
                            best_combo = [l1_c, l2_c, l3_c]

        if best_combo is not None:
            return best_combo
        else:
            return best_fallback_duo

    def classify_line_type(self, line, paint_mask):
        height, width = paint_mask.shape
        m = line['m']
        b = line['b']

        y_bottom = height - 1
        y_top = int(height * 0.3)

        hits = 0
        total_evaluated = 0
        search_margin = 10

        for y in range(y_bottom, y_top, -2):
            x = int(m * y + b)

            if x < 0 or x >= width:
                continue

            total_evaluated += 1

            x_start = max(0, x - search_margin)
            x_end = min(width, x + search_margin + 1)

            if np.any(paint_mask[y, x_start:x_end] > 0):
                hits += 1

        if total_evaluated == 0:
            return 'solid'

        fill_ratio = hits / total_evaluated

        if fill_ratio > 0.60:
            return 'solid'
        else:
            return 'dashed'

    def transform_to_classical(self, lines, original_height):
        classical_lines = []
        crop_h = int(original_height * self.config['crop_top'])

        for line in lines:
            m_inv = line['m']
            b_inv_crop = line['b']

            b_inv_orig = b_inv_crop - (m_inv * crop_h)

            m_safe = m_inv if abs(m_inv) > 1e-5 else 1e-5

            a = 1.0 / m_safe
            b = -b_inv_orig / m_safe

            classical_lines.append({
                'a': a,
                'b': b,
                'weight': line.get('weight', 0),
                'type': line.get('type', 'solid')
            })

        return classical_lines

    def draw_lines(self, original_img, classical_lines):
        result_img = original_img.copy()
        h_orig = original_img.shape[0]

        crop_h = int(h_orig * self.config['crop_top'])

        for line in classical_lines:
            a = line['a']
            b = line['b']

            y1_orig = h_orig
            x1 = int((y1_orig - b) / a)

            y2_orig = crop_h
            x2 = int((y2_orig - b) / a)

            if line.get('type') == 'dashed':
                color = (0, 0, 255)
            else:
                color = (0, 255, 0)

            thickness = 4

            cv2.line(result_img, (x1, y1_orig), (x2, y2_orig), color, thickness)

        return result_img

    def detect(self, image):
        self.scale_params(image)
        cropped_img = self.crop_image(image)
        mask = self.get_adaptive_mask(cropped_img)
        math_lines = self.get_math_lines(mask)

        for line in math_lines:
            line['type'] = self.classify_line_type(line, mask)

        final_lines = self.transform_to_classical(math_lines, image.shape[0])
        return final_lines