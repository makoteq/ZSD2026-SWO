import os
import cv2
import time
import csv
import argparse
import psutil
import numpy as np
import ultralytics
from collections import defaultdict
import supervision as sv
from ultralytics import YOLO

def get_roi_mask(frame):
    points = []
    temp_frame = frame.copy()

    def click_event(event, x, y, flags, params):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            cv2.circle(temp_frame, (x, y), 5, (0, 0, 255), -1)
            if len(points) >= 2:
                cv2.line(temp_frame, points[-2], points[-1], (0, 0, 255), 2)
            cv2.imshow("Define ROI", temp_frame)

    cv2.imshow("Define ROI", temp_frame)
    cv2.setMouseCallback("Define ROI", click_event)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == 13 or key == 10:  # Enter
            break
        elif key == ord('r'):
            points = []
            temp_frame = frame.copy()
            cv2.imshow("Define ROI", temp_frame)

    cv2.destroyWindow("Define ROI")

    if len(points) > 2:
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        pts = np.array(points, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
        return mask, pts
    else:
        print("ERROR, no valid ROI selected. Analysis will include the entire frame.")
        return None, None

def get_rpi_temperature():
    try:
        res = os.popen("vcgencmd measure_temp").readline()
        return float(res.replace("temp=", "").replace("'C\n", ""))
    except Exception:
        return 0.0 

def parse_args():
    parser = argparse.ArgumentParser(description="YOLO Benchmarking Script with ROI Selection")
    parser.add_argument('--model', type=str, required=True, help="Model path") 
    parser.add_argument('--video', type=str, default=r"CARLA_test.mp4", help="Video file path")
    parser.add_argument('--slice_size', type=int, default=320, help="Input size for YOLO (imgsz)")
    parser.add_argument('--output_csv', type=str, default="yolo_metrics_output.csv", help="Output CSV file for metrics")
    parser.add_argument('--output_img', type=str, default="yolo_last_frame.jpg", help="Image file for last frame")
    parser.add_argument('--show', type=int, default=0, help="Display video: 1=Yes, 0=No")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"BENCHMARK YOLO - Model: {os.path.basename(args.model)}, Slice Size: {args.slice_size}")
    if not os.path.exists(args.video):
        print(f"ERROR, video file not found: {args.video}")
        return

    cap = cv2.VideoCapture(args.video)
    success, first_frame = cap.read()
    if not success:
        print("ERROR, can not read video.")
        return

    roi_mask, roi_points = get_roi_mask(first_frame)
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    try:
        model = YOLO(args.model)
    except Exception as e:
        print(f"ERROR, can not load model: {e}")
        return

    tracker = sv.ByteTrack(track_activation_threshold=0.15, lost_track_buffer=120)
    
    fps_video = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('traffic_yolo_tracked.mp4', fourcc, fps_video, 
                         (int(cap.get(3)), int(cap.get(4))))

    file_exists = os.path.isfile(args.output_csv)
    csv_file = open(args.output_csv, mode='a', newline='')
    fieldnames = ['Model', 'Slice_Size', 'Frame', 'Total_FPS', 'Inference_ms', 
                  'Objects_On_Frame', 'Unique_Objects_Total', 'Avg_Confidence', 
                  'CPU_Percent', 'RAM_Percent', 'Temp_C']
    writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()

    unique_vehicle_ids = set()
    track_history = defaultdict(lambda: [])
    frame_count = 0
    warmup_frames = 30 
    last_annotated_frame = None
    
    print(f"Warmup frames: {warmup_frames}")

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break
            
            frame_count += 1
            t_start = time.time()

            if roi_mask is not None:
                masked_frame = cv2.bitwise_and(frame, frame, mask=roi_mask)
            else:
                masked_frame = frame

            results = model(
                source=masked_frame,
                imgsz=args.slice_size,
                iou=0.45,
                conf=0.2,
                verbose=False,
                classes=[0]
            )

            t_end = time.time()
            inference_ms = (t_end - t_start) * 1000
            real_fps = 1.0 / (t_end - t_start) if (t_end - t_start) > 0 else 0.0

            xyxy, confidences, class_ids = [], [], []
            boxes = results[0].boxes
            
            if len(boxes) > 0:
                for box in boxes:
                    b = box.xyxy[0].cpu().numpy()
                    xyxy.append([b[0], b[1], b[2], b[3]])
                    confidences.append(float(box.conf[0]))
                    class_ids.append(int(box.cls[0]))

            detections = sv.Detections(
                xyxy=np.array(xyxy) if xyxy else np.empty((0, 4)),
                confidence=np.array(confidences) if confidences else np.array([]),
                class_id=np.array(class_ids) if class_ids else np.array([])
            )

            detections = tracker.update_with_detections(detections)
            
            annotated_frame = frame.copy()
            
            if roi_points is not None:
                cv2.polylines(annotated_frame, [roi_points], isClosed=True, color=(0, 255, 255), thickness=2)

            if detections.tracker_id is not None:
                current_ids = detections.tracker_id.tolist()
                unique_vehicle_ids.update(current_ids)
                
                for bbox, track_id in zip(detections.xyxy, detections.tracker_id):
                    x1, y1, x2, y2 = bbox.astype(int)
                    
                    track = track_history[track_id]
                    track.append((float((x1+x2)/2), float(y2)))  
                    if len(track) > 30: track.pop(0)
                    
                    points = np.array(track).astype(np.int32).reshape((-1, 1, 2))
                    cv2.polylines(annotated_frame, [points], isClosed=False, color=(0, 255, 0), thickness=2)

                    cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
                    cv2.putText(annotated_frame, f"ID:{track_id}", (x1, y1 - 10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

            if frame_count > warmup_frames:
                writer.writerow({
                    'Model': os.path.basename(args.model),
                    'Slice_Size': args.slice_size,
                    'Frame': frame_count - warmup_frames,
                    'Total_FPS': round(real_fps, 2),
                    'Inference_ms': round(inference_ms, 2),
                    'Objects_On_Frame': len(xyxy),
                    'Unique_Objects_Total': len(unique_vehicle_ids),
                    'Avg_Confidence': round(float(np.mean(confidences)), 3) if confidences else 0.0,
                    'CPU_Percent': psutil.cpu_percent(),
                    'RAM_Percent': psutil.virtual_memory().percent,
                    'Temp_C': get_rpi_temperature() 
                })

            if args.show == 1:
                cv2.putText(annotated_frame, f"Unikalne auta: {len(unique_vehicle_ids)}", (10, 90), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                cv2.imshow("YOLO Tracking", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'): break

            out.write(annotated_frame)
            if frame_count == 500:
                last_annotated_frame = annotated_frame.copy()
            
            if frame_count % 10 == 0:
                print(f"Klatka {frame_count} | Unique cars: {len(unique_vehicle_ids)} | FPS: {real_fps:.1f}")

    except KeyboardInterrupt:
        print("\nInterrupted manually.")
    finally:
        cap.release()
        out.release()
        csv_file.close()
        cv2.destroyAllWindows()

        if last_annotated_frame is not None:
            output_path = os.path.join("YOLObenchmarks",args.output_img)
            cv2.imwrite(output_path, last_annotated_frame)

        print("\n--- Test summary --- ")
        print(f"Total frames processed: {max(0, frame_count - warmup_frames)}")
        print(f"Actual number of unique vehicles: {len(unique_vehicle_ids)}")
        print(f"Data saved to: {args.output_csv}")

if __name__ == "__main__":
    main()