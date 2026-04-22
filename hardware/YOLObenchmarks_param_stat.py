import cv2
import time
import csv
import argparse
import os
import psutil
import numpy as np
from ultralytics import YOLO

def get_rpi_temperature():
    try:
        res = os.popen("vcgencmd measure_temp").readline()
        return float(res.replace("temp=", "").replace("'C\n", ""))
    except Exception:
        return 0.0

def parse_args():
    parser = argparse.ArgumentParser(description="Tracking YOLOv8 Edge TPU z monitoringiem sprzętu")
    parser.add_argument('--model', type=str, required=True, help="Ścieżka do modelu .tflite")
    parser.add_argument('--video', type=str, default="CARLA_test.mp4", help="Ścieżka do wideo")
    parser.add_argument('--imgsz', type=int, required=True, help="Rozmiar obrazu (np. 320, 640)")
    parser.add_argument('--output_csv', type=str, default="tracking_metrics.csv", help="Plik CSV")
    parser.add_argument('--output_img', type=str, default="wynik_detekcji.jpg", help="Nazwa zapisanego zdjęcia")
    parser.add_argument('--show', action='store_true', help="Pokaż podgląd na żywo")
    return parser.parse_args()

def main():
    args = parse_args()
    
    try:
        model = YOLO(args.model, task='detect')
    except Exception as e:
        print(f"ERROR, unable to load model: {e}")
        return

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"ERROR, unable to open video: {args.video}")
        return

    unique_vehicle_ids = set()
    metrics_data = []
    frame_count = 0
    warmup_frames = 30  # Ignorujemy pierwsze 30 klatek przy zapisie metryk
    total_start_time = time.time()
    last_annotated_frame = None

    print(f"Warmup: {warmup_frames}")

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            frame_count += 1
            t_start = time.time()

            # Inferencja i Tracking
            results = model.track(
                source=frame, 
                imgsz=args.imgsz,
                iou=0.45,
                conf=0.2, 
                persist=True, 
                verbose=False, 
                classes=[0] 
            )

            t_end = time.time()
            real_fps = 1.0 / (t_end - t_start) if (t_end - t_start) > 0 else 0.0

            res = results[0]
            boxes = res.boxes
            res.names = {0: "Car"}

            # Logika unikalnych ID
            if boxes is not None and boxes.id is not None:
                current_ids = boxes.id.int().cpu().tolist()
                unique_vehicle_ids.update(current_ids)

            num_detections = len(boxes)
            avg_confidence = float(np.mean(boxes.conf.cpu().numpy())) if num_detections > 0 else 0.0

            annotated_frame = res.plot()
            cv2.putText(annotated_frame, f"Unique cars: {len(unique_vehicle_ids)}", (10, 60), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            last_annotated_frame = annotated_frame

            if frame_count > warmup_frames:
                speed = res.speed
                inference_ms = speed.get('inference', 0.0)

                metrics_data.append({
                    'Model': os.path.basename(args.model),
                    'Slice_Size': args.imgsz, # Mapujemy imgsz jako Slice_Size dla spójności
                    'Frame': frame_count - warmup_frames,
                    'Total_FPS': round(real_fps, 2),
                    'Inference_ms': round(inference_ms, 2),
                    'Objects_On_Frame': num_detections,
                    'Unique_Objects_Total': len(unique_vehicle_ids),
                    'Avg_Confidence': round(avg_confidence, 3),
                    'CPU_Percent': psutil.cpu_percent(),
                    'RAM_Percent': psutil.virtual_memory().percent,
                    'Temp_C': get_rpi_temperature()
                })

            if args.show:
                cv2.imshow("YOLO Tracking", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            if frame_count % 100 == 0:
                print(f"Klatka {frame_count} | Unikalne auta: {len(unique_vehicle_ids)} | FPS: {real_fps:.1f} | Temp: {get_rpi_temperature()}°C")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        
        if last_annotated_frame is not None:
            cv2.imwrite(args.output_img, last_annotated_frame)
            print(f"\nZapisano klatkę wynikową do: {args.output_img}")

        # Zapis do pliku CSV
        if metrics_data:
            with open(args.output_csv, mode='a', newline='') as f:
                writer = csv.DictWriter(f, fieldnames=metrics_data[0].keys())
                # Zapisz nagłówki tylko jeśli plik jest pusty
                if f.tell() == 0: 
                    writer.writeheader()
                writer.writerows(metrics_data)

            print(f"Rzeczywista liczba unikalnych pojazdów: {len(unique_vehicle_ids)}")
            print(f"Zebrane metryki (odrzucono {warmup_frames} klatek rozgrzewki) zapisano w: {args.output_csv}")

if __name__ == "__main__":
    main()