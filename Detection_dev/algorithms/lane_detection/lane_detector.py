from ultralytics import YOLO
import cv2
import numpy as np
from itertools import combinations


class LaneDetector:
    def __init__(self, model_path, config=None):
        self.yolo = YOLO(model_path)

        self.config = {
            'yolo_conf': 0.25,
            'adapt_c': -15,
            'vertical_ratio': 0.5,
            'num_segments': 6,
            'local_density_thresh': 0.20,
            'min_active_segments': 2,
            'post_aspect_ratio': 3.0,
            'post_min_solidity': 0.6,
            'max_green_inside': 0.05,
            'max_green_halo': 0.30,
            'green_h_range': (35, 85),
            'green_s_min': 40,
            'display_width': 500,
            'cluster_m_tol': 0.10,
            'cluster_b_tol_pct': 0.04,
            'min_line_coverage': 0.02,
            'adapt_block_size_base': 81,
            'hough_thresh_base': 20,
            'hough_min_len_base': 30,
            'hough_max_gap_base': 100,
            'post_min_area_base': 20,
            'halo_thickness_base': 15,
            'search_radius_base': 15,
            'y_step_base': 5,
            'max_debug_width': 1600,
            'max_debug_height': 600
        }

        if config:
            self.config.update(config)

    def scale_parameters(self, height, width):
        scale_factor = width / 1280.0

        def make_odd(val):
            v = int(val)
            return v if v % 2 != 0 else max(3, v + 1)

        return {
            'adapt_block_size': make_odd(self.config['adapt_block_size_base'] * scale_factor),
            'hough_thresh': max(5, int(self.config['hough_thresh_base'] * scale_factor)),
            'hough_min_len': max(10, int(self.config['hough_min_len_base'] * scale_factor)),
            'hough_max_gap': max(20, int(self.config['hough_max_gap_base'] * scale_factor)),
            'post_min_area': max(5, int(self.config['post_min_area_base'] * (scale_factor ** 2))),
            'halo_thickness': max(3, int(self.config['halo_thickness_base'] * scale_factor)),
            'y_step': max(2, int(self.config['y_step_base'] * (height / 720.0))),
            'search_r': max(3, int(self.config['search_radius_base'] * scale_factor)),
            'line_thick': max(3, int(6 * scale_factor)),
            'font_scale': 0.7 * (width / 1000.0),
            'road_margin': max(5, int(5 * scale_factor))
        }

    def calculate_line_coverage(self, m_inv, b_inv, height, width, reference_mask, scaled_params, horizon_y):
        hits, total_points = 0, 0
        search_r = scaled_params['search_r']
        for y in range(height - 1, horizon_y, -scaled_params['y_step']):
            total_points += 1
            x_ideal = int(m_inv * y + b_inv)
            if 0 <= x_ideal < width:
                x_start, x_end = max(0, x_ideal - search_r), min(width, x_ideal + search_r + 1)
                if np.any(reference_mask[y, x_start:x_end] > 0):
                    hits += 1
        return hits / total_points if total_points > 0 else 0

    def add_text(self, img, text):
        cv2.putText(img, text, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3)
        cv2.putText(img, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        return img

    def build_mosaic(self, images, titles, grid_size=(3, 3)):
        rows, cols = grid_size
        grid_rows = []
        target_width = self.config['display_width']
        for r in range(rows):
            row_images = []
            for c in range(cols):
                idx = r * cols + c
                if idx < len(images):
                    img = images[idx]
                    if len(img.shape) == 2:
                        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    resized = cv2.resize(img, (target_width, int(img.shape[0] * (target_width / img.shape[1]))))
                    row_images.append(self.add_text(resized, titles[idx]))
                else:
                    row_images.append(np.zeros((300, target_width, 3), dtype=np.uint8))
            grid_rows.append(np.hstack(row_images))
        return np.vstack(grid_rows)

    def _get_road_mask(self, img, height, width, scaled_params):
        results = self.yolo.predict(img, conf=self.config['yolo_conf'], verbose=False)
        road_mask = np.full((height, width), 255, dtype=np.uint8)

        if results[0].masks is not None:
            road_mask = cv2.resize(results[0].masks.data[0].cpu().numpy(), (width, height))
            road_mask = (road_mask * 255).astype(np.uint8)
            margin = scaled_params['road_margin']
            kernel = np.ones((margin, margin), np.uint8)
            road_mask = cv2.erode(road_mask, kernel, iterations=1)

        y_horizon = np.min(np.where(road_mask > 0)[0]) if np.any(road_mask > 0) else 0
        return road_mask, y_horizon

    def _get_paint_mask(self, img, road_mask, scaled_params, debug_dict=None):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        lower_yellow = np.array([15, 50, 120])
        upper_yellow = np.array([35, 255, 255])
        yellow_mask = cv2.inRange(hsv, lower_yellow, upper_yellow)

        road_pixels = gray[road_mask > 0]

        if len(road_pixels) == 0:
            white_mask = np.zeros_like(gray)
        else:
            mean_brightness = np.mean(road_pixels)

            if mean_brightness > 200:
                p98 = np.percentile(road_pixels, 98)
                thresh_val = p98 - 6
                _, paint_mask = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY)

                kernel_width = max(21, int(scaled_params['adapt_block_size'] * 0.4))
                horizontal_kernel = np.ones((1, kernel_width), np.uint8)

                glare_blobs = cv2.morphologyEx(paint_mask, cv2.MORPH_OPEN, horizontal_kernel)
                lines_only = cv2.bitwise_xor(paint_mask, glare_blobs)

                white_mask = cv2.morphologyEx(lines_only, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            else:
                blur = cv2.GaussianBlur(gray, (11, 11), 0)
                white_mask = cv2.adaptiveThreshold(
                    blur, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,
                    scaled_params['adapt_block_size'], self.config['adapt_c']
                )

        combined_paint_mask = cv2.bitwise_or(white_mask, yellow_mask)

        if debug_dict is not None:
            debug_dict['adaptive_raw'] = combined_paint_mask

        return cv2.bitwise_and(combined_paint_mask, road_mask)
    def _get_verified_mask(self, paint_mask, scaled_params, height, width):
        verified_mask = np.zeros_like(paint_mask)
        lines = cv2.HoughLinesP(
            paint_mask, 1, np.pi / 180, scaled_params['hough_thresh'],
            minLineLength=scaled_params['hough_min_len'], maxLineGap=scaled_params['hough_max_gap']
        )

        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(y2 - y1) > abs(x2 - x1) * self.config['vertical_ratio']:
                    m, b = (x2 - x1) / (y2 - y1), x1 - ((x2 - x1) / (y2 - y1)) * y1
                    active_segments, seg_h = 0, height // self.config['num_segments']
                    for s in range(self.config['num_segments']):
                        y_s, y_e = s * seg_h, (s + 1) * seg_h
                        hits, pts = 0, 0
                        for y_seg in range(y_s, y_e, 3):
                            x_seg = int(m * y_seg + b)
                            if 0 <= x_seg < width:
                                pts += 1
                                if paint_mask[y_seg, x_seg] > 0:
                                    hits += 1
                        if pts > 0 and (hits / pts) > self.config['local_density_thresh']:
                            active_segments += 1
                    if active_segments >= self.config['min_active_segments']:
                        cv2.line(verified_mask, (int(m * 0 + b), 0), (int(m * height + b), height), 255,
                                 max(5, int(12 * (width / 1280.0))))
        return verified_mask

    def _filter_blobs(self, img, paint_mask, verified_mask, scaled_params, debug_dict=None):
        raw_lines = cv2.bitwise_and(paint_mask, verified_mask)
        shape_filtered = np.zeros_like(raw_lines)
        clean_lanes = np.zeros_like(raw_lines)

        hsv_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        green_mask = cv2.inRange(
            hsv_img,
            np.array([self.config['green_h_range'][0], self.config['green_s_min'], 20]),
            np.array([self.config['green_h_range'][1], 255, 255])
        )

        contours, _ = cv2.findContours(raw_lines, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area < scaled_params['post_min_area']:
                continue

            rect = cv2.minAreaRect(cnt)
            (x_rect, y_rect), (w_rect, h_rect), _ = rect
            if w_rect == 0 or h_rect == 0:
                continue

            aspect_ratio = max(w_rect, h_rect) / min(w_rect, h_rect)

            if aspect_ratio < self.config['post_aspect_ratio']:
                continue

            hull = cv2.convexHull(cnt)
            hull_area = cv2.contourArea(hull)

            if hull_area > 0:
                solidity = area / hull_area

                min_req_solidity = 0.2 if aspect_ratio > 8.0 else self.config['post_min_solidity']

                if solidity < min_req_solidity:
                    continue

            cv2.drawContours(shape_filtered, [cnt], -1, 255, -1)

            inside_mask = np.zeros_like(raw_lines)
            cv2.drawContours(inside_mask, [cnt], -1, 255, -1)

            if cv2.countNonZero(inside_mask) > 0:
                green_ratio = cv2.countNonZero(cv2.bitwise_and(green_mask, inside_mask)) / cv2.countNonZero(inside_mask)
                if green_ratio <= self.config['max_green_inside']:
                    cv2.drawContours(clean_lanes, [cnt], -1, 255, -1)

        if debug_dict is not None:
            debug_dict['raw_lines'] = raw_lines
            debug_dict['shape_filtered'] = shape_filtered

        return clean_lanes

    def _get_raw_candidates(self, clean_lanes, scaled_params, height, width, y_horizon):
        clean_contours, _ = cv2.findContours(clean_lanes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        raw_candidates = []

        for cnt in clean_contours:
            if len(cnt) < 2:
                continue


            blob_mask = np.zeros_like(clean_lanes)
            cv2.drawContours(blob_mask, [cnt], -1, 255, -1)

            internal_pts = cv2.findNonZero(blob_mask)

            if internal_pts is None or len(internal_pts) < 10:
                continue

            [vx, vy, x0, y0] = cv2.fitLine(internal_pts, cv2.DIST_L2, 0, 0.01, 0.01)

            if vy == 0:
                continue

            m_inv, b_inv = (vx / vy)[0], (x0 - (vx / vy) * y0)[0]

            x_bot = m_inv * height + b_inv
            x_top = m_inv * y_horizon + b_inv

            if 0 <= x_bot <= width and 0 <= x_top <= width:
                coverage = self.calculate_line_coverage(m_inv, b_inv, height, width, clean_lanes, scaled_params,
                                                        y_horizon)
                if coverage >= self.config['min_line_coverage']:
                    raw_candidates.append({
                        'm': m_inv, 'b': b_inv, 'x_bot': x_bot, 'cov': coverage, 'abs_m': abs(m_inv)
                    })

        return raw_candidates

    def _filter_unique_lines(self, raw_candidates, width):
        unique_lines = []
        for cand in sorted(raw_candidates, key=lambda x: x['cov'], reverse=True):
            is_dup = False
            for ul in unique_lines:
                if abs(cand['m'] - ul['m']) < 0.05 and abs(cand['x_bot'] - ul['x_bot']) < width * 0.05:
                    is_dup = True
                    break
            if not is_dup:
                unique_lines.append(cand)
        return unique_lines

    def _find_best_group(self, unique_lines, height, width):
        valid_groups = []
        min_lane_width = width * 0.10
        max_lane_width = width * 0.80

        for n in [3, 2]:
            if len(unique_lines) < n:
                continue

            for combo in combinations(unique_lines, n):
                combo = sorted(combo, key=lambda x: x['x_bot'])
                dists = [combo[i + 1]['x_bot'] - combo[i]['x_bot'] for i in range(len(combo) - 1)]
                if any(d < min_lane_width or d > max_lane_width for d in dists):
                    continue

                intersect_points = []
                valid_intersection = True

                for i, j in combinations(range(len(combo)), 2):
                    l1, l2 = combo[i], combo[j]
                    dm = l1['m'] - l2['m']

                    if abs(dm) < 0.001:
                        intersect_points.append((l1['x_bot'], -100000))
                    else:
                        y_int = (l2['b'] - l1['b']) / dm
                        x_int = l1['m'] * y_int + l1['b']

                        if y_int > height * 0.8:
                            valid_intersection = False
                            break
                        intersect_points.append((x_int, y_int))

                if not valid_intersection:
                    continue

                pts_x = [p[0] for p in intersect_points]
                x_spread = max(pts_x) - min(pts_x) if len(pts_x) > 0 else 0
                symmetry_err = abs(dists[0] - dists[1]) if len(dists) == 2 else 0
                n_penalty = 0 if n == 3 else width * 2.0
                total_score = x_spread + (symmetry_err * 0.5) + n_penalty

                valid_groups.append({
                    'combo': combo,
                    'score': total_score,
                    'x_spread': x_spread
                })

        best_group = []
        if valid_groups:
            valid_groups.sort(key=lambda g: g['score'])
            best_group = valid_groups[0]['combo']

        if not best_group and unique_lines:
            best_group = [max(unique_lines, key=lambda x: x['cov'])]

        return best_group

    def _display_debug(self, img, road_mask, paint_mask, verified_mask, clean_lanes, best_group, scaled_params,
                       y_horizon, debug_dict):
        height = img.shape[0]

        final_overlay = img.copy()
        for i, lane in enumerate(best_group):
            x_b, x_t = int(lane['x_bot']), int(lane['m'] * y_horizon + lane['b'])
            cv2.line(final_overlay, (x_b, height), (x_t, y_horizon), (0, 0, 255), scaled_params['line_thick'])
            cv2.putText(final_overlay, f"L{i + 1}", (x_b - 20, height - 20), cv2.FONT_HERSHEY_SIMPLEX,
                        scaled_params['font_scale'], (0, 0, 255), 2)

        res_final = cv2.addWeighted(img, 0.7, final_overlay, 0.8, 0)
        res_final[clean_lanes > 0] = [0, 255, 0]

        debug_images = [
            img.copy(),
            road_mask.copy(),
            debug_dict.get('adaptive_raw', np.zeros_like(road_mask)),
            paint_mask.copy(),
            verified_mask.copy(),
            debug_dict.get('raw_lines', np.zeros_like(road_mask)),
            debug_dict.get('shape_filtered', np.zeros_like(road_mask)),
            clean_lanes.copy(),
            res_final
        ]

        debug_titles = [
            "1. Original",
            "2. Yolo mask",
            "3. Adaptive mask",
            "4. Combined mask",
            "5. Vertical Density Filter (Hough)",
            "6. Candidates",
            "7. Shape Filter",
            "8. Color Filter",
            "9. Result"
        ]

        mosaic = self.build_mosaic(debug_images, debug_titles, grid_size=(3, 3))

        max_w = self.config['max_debug_width']
        max_h = self.config['max_debug_height']
        h, w = mosaic.shape[:2]

        if h > max_h or w > max_w:
            scale = min(max_w / w, max_h / h)
            mosaic = cv2.resize(mosaic, (int(w * scale), int(h * scale)))

        cv2.namedWindow("Debug Panel 3x3", cv2.WINDOW_NORMAL)
        cv2.imshow("Debug Panel 3x3", mosaic)

    def process_image(self, original_img, debug=False):
        orig_h, orig_w = original_img.shape[:2]
        crop_y = orig_h // 2
        img = original_img[crop_y:orig_h, :].copy()
        height, width = img.shape[:2]
        scaled_params = self.scale_parameters(height, width)

        debug_dict = {} if debug else None

        road_mask, y_horizon = self._get_road_mask(img, height, width, scaled_params)
        paint_mask = self._get_paint_mask(img, road_mask, scaled_params, debug_dict)
        verified_mask = self._get_verified_mask(paint_mask, scaled_params, height, width)
        clean_lanes = self._filter_blobs(img, paint_mask, verified_mask, scaled_params, debug_dict)

        raw_candidates = self._get_raw_candidates(clean_lanes, scaled_params, height, width, y_horizon)
        unique_lines = self._filter_unique_lines(raw_candidates, width)
        best_group = self._find_best_group(unique_lines, height, width)

        if debug:
            self._display_debug(img, road_mask, paint_mask, verified_mask, clean_lanes, best_group, scaled_params,
                                y_horizon, debug_dict)

        return best_group

    def draw_lanes(self, original_img, lines):
        result_img = original_img.copy()
        orig_h, orig_w = original_img.shape[:2]
        crop_y = orig_h // 2

        scaled_params = self.scale_parameters(orig_h - crop_y, orig_w)

        for i, lane in enumerate(lines):
            x_top = int(lane['m'] * (0 - crop_y) + lane['b'])
            x_bot = int(lane['m'] * (orig_h - crop_y) + lane['b'])

            cv2.line(result_img, (x_bot, orig_h), (x_top, 0), (0, 255, 0), scaled_params['line_thick'])

            cv2.putText(
                result_img,
                f"L{i + 1}",
                (x_bot - 20, orig_h - 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                scaled_params['font_scale'],
                (0, 255, 0),
                2
            )

        return result_img