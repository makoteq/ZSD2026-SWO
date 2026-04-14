import cv2
import numpy as np


class LaneDetector:
    def __init__(self):
        # Background and line quality thresholds
        self.bg_tolerance = 140
        self.max_bg_sat = 60
        self.min_contrast = 15
        self.min_fill = 0.05

        # Threshold to classify a line as solid vs dashed
        self.solid_fill_threshold = 0.40

        self.low_white = np.array([0, 0, 190])
        self.high_white = np.array([180, 30, 255])
        self.low_yellow = np.array([10, 35, 70])
        self.high_yellow = np.array([50, 255, 255])

    def detect(self, img):
        if img is None:
            return []

        orig_h, orig_w = img.shape[:2]
        crop_y = orig_h // 2

        # Analyze only the bottom half
        bottom_img = img[crop_y:orig_h, :].copy()
        h, w = bottom_img.shape[:2]
        scale = w / 1280.0

        # Dynamic parameters
        params = {
            'y_step': max(2, int(6 * (h / 720.0))),
            'search_r': max(3, int(15 * scale)),
            'check_r': max(6, int(30 * scale)),
            'min_len': max(50, int(100 * scale)),
            'max_gap': max(50, int(250 * scale)),
            'dup_dist': int(w * 0.12),
            'max_line_width': int(50 * scale)
        }

        hsv_img = cv2.cvtColor(bottom_img, cv2.COLOR_BGR2HSV)
        roi_mask = cv2.inRange(hsv_img, np.array([0, 0, 0]), np.array([180, 50, 255]))
        kernel = np.ones((max(5, int(25 * scale)), max(5, int(25 * scale))), np.uint8)
        roi_patched = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, kernel)

        gray = cv2.cvtColor(bottom_img, cv2.COLOR_BGR2GRAY)
        blur_size = max(3, int(5 * scale) | 1)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (blur_size, blur_size), 0), 30, 100)
        edges_roi = cv2.bitwise_and(edges, roi_patched)

        # Hough Transform
        raw_lines = cv2.HoughLinesP(
            edges_roi, 1, np.pi / 180, 20,
            minLineLength=params['min_len'], maxLineGap=params['max_gap']
        )

        vertical_lines = []
        if raw_lines is not None:
            for line in raw_lines:
                x1, y1, x2, y2 = line[0]
                if abs(x2 - x1) > abs(y2 - y1):
                    continue
                vertical_lines.append((x1, y1, x2, y2))

        mask_w = cv2.inRange(hsv_img, self.low_white, self.high_white)
        mask_y = cv2.inRange(hsv_img, self.low_yellow, self.high_yellow)
        color_mask = cv2.bitwise_or(mask_w, mask_y)

        scored_lines = []
        for x1, y1, x2, y2 in vertical_lines:
            if x1 != x2:
                m = (y2 - y1) / (x2 - x1)
                b_inter = y1 - m * x1
                x_bot = int((h - b_inter) / m)
                x_top = int((-b_inter) / m)
            else:
                x_bot, x_top = x1, x1

            score, fill = self._evaluate_line(x_bot, x_top, h, w, color_mask, bottom_img, hsv_img, gray, params)

            if score >= 0.70:
                scored_lines.append({
                    'x_bot': x_bot, 'x_top': x_top,
                    'score': score, 'fill': fill
                })

        scored_lines.sort(key=lambda x: x['score'], reverse=True)
        final_lanes = []

        for line in scored_lines:
            is_dup = False
            for selected in final_lanes:
                dist_bot = abs(line['x_bot'] - selected['x_bot'])
                x_mid = line['x_bot'] + (line['x_top'] - line['x_bot']) * 0.5
                s_mid = selected['x_bot'] + (selected['x_top'] - selected['x_bot']) * 0.5
                dist_mid = abs(x_mid - s_mid)

                if dist_bot < params['dup_dist'] or dist_mid < (params['dup_dist'] * 0.6):
                    is_dup = True
                    break

            if not is_dup:
                final_lanes.append(line)
            if len(final_lanes) == 3:
                break

        results = []
        for lane in final_lanes:
            pt1 = (lane['x_bot'], orig_h)
            pt2 = (lane['x_top'], crop_y)

            dx = pt2[0] - pt1[0]
            dy = pt2[1] - pt1[1]

            if dx == 0:
                a, b = float('inf'), pt1[0]
            else:
                a = dy / dx
                b = pt1[1] - a * pt1[0]

            line_type = "solid" if lane['fill'] >= self.solid_fill_threshold else "dashed"

            results.append({
                'a': a, 'b': b, 'type': line_type,
                'score': lane['score'], 'pts': (pt1, pt2)
            })

        return sorted(results, key=lambda r: r['pts'][0][0])

    def draw_lines(self, img, lines):
        result_img = img.copy()
        thick = max(2, int(5 * (img.shape[1] / 1280.0)))

        for line in lines:
            pt1, pt2 = line['pts']
            color = (0, 255, 0) if line['type'] == 'solid' else (0, 0, 255)
            cv2.line(result_img, pt1, pt2, color, thick)

        return result_img

    def _evaluate_line(self, x_bot, x_top, h, w, mask, img_bgr, img_hsv, img_gray, params):
        valid_hits = 0
        paint_hits = 0
        total_span = 0

        for y in range(h - 1, 0, -params['y_step']):
            total_span += 1
            p_h = y / h
            curr_search = max(2, int(params['search_r'] * p_h))
            curr_check = max(4, int(params['check_r'] * p_h))

            x_ideal = int(x_bot + (x_top - x_bot) * (1.0 - p_h))
            if x_ideal - curr_check < 0 or x_ideal + curr_check >= w:
                continue

            local_x = x_ideal
            max_g = -1
            for dx in range(-curr_search, curr_search + 1):
                cx = x_ideal + dx
                if 0 <= cx < w and img_gray[y, cx] > max_g:
                    max_g = img_gray[y, cx]
                    local_x = cx

            xc = local_x
            if xc - curr_check < 0 or xc + curr_check >= w:
                continue

            g_C = int(img_gray[y, xc])
            g_L = int(img_gray[y, xc - curr_check])
            g_R = int(img_gray[y, xc + curr_check])

            is_contrast = (g_C > g_L + self.min_contrast) and (g_C > g_R + self.min_contrast)
            is_valid_col = mask[y, xc] > 0

            if is_contrast and is_valid_col:
                w_left = 0
                while (xc - w_left >= 0) and (img_gray[y, xc - w_left] > g_C - 25):
                    w_left += 1

                w_right = 0
                while (xc + w_right < w) and (img_gray[y, xc + w_right] > g_C - 25):
                    w_right += 1

                max_allowed_width = max(8, int(params['max_line_width'] * p_h))

                if (w_left + w_right) > max_allowed_width:
                    continue

                paint_hits += 1

                col_L = img_bgr[y, xc - curr_check].astype(np.int32)
                col_R = img_bgr[y, xc + curr_check].astype(np.int32)
                sat_L = img_hsv[y, xc - curr_check, 1]
                sat_R = img_hsv[y, xc + curr_check, 1]

                is_asphalt_L = (sat_L <= self.max_bg_sat)
                is_asphalt_R = (sat_R <= self.max_bg_sat)
                is_symmetric = np.sum(np.abs(col_L - col_R)) <= self.bg_tolerance

                is_valid_bg = False
                if is_asphalt_L and is_asphalt_R and is_symmetric:
                    is_valid_bg = True
                elif is_asphalt_L or is_asphalt_R:
                    if (g_C > g_L + 30) and (g_C > g_R + 30):
                        is_valid_bg = True

                if is_valid_bg:
                    valid_hits += 1

        fill_ratio = paint_hits / total_span if total_span > 0 else 0
        if fill_ratio < self.min_fill:
            return 0.0, fill_ratio

        quality = valid_hits / paint_hits if paint_hits > 0 else 0
        density = min(1.0, fill_ratio / 0.15)

        return quality * density, fill_ratio