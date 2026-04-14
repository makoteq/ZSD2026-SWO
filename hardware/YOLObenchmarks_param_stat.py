import cv2
import time
import csv
import argparse
import os
import numpy as np
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser(description="Testowanie YOLOv8 na Google Coral Edge TPU")
    parser.add_argument('--model', type=str, required=True, help="Ścieżka do modelu (np. yolov8n_320_edgetpu.tflite)")
    parser.add_argument('--video', type=str, default=r"traffic.mp4", help="Ścieżka do wideo")
    parser.add_argument('--imgsz', type=int, required=True, help="Rozmiar wejściowy obrazu (np. 240, 320, 512, 640)")
    parser.add_argument('--output_csv', type=str, default=f"YOLObenchmarks/256_metrics_output.csv", help="Plik wyjściowy dla metryk")
    parser.add_argument('--show', action='store_true', help="Wyświetlaj wideo podczas inferencji (obniża FPS)")
    return parser.parse_args()

def main():
    args = parse_args()
    
    print(f"Ładowanie modelu: {args.model} z rozdzielczością {args.imgsz}x{args.imgsz}")
    
    try:
        model = YOLO(args.model, task='detect')
    except Exception as e:
        print(f"Błąd podczas ładowania modelu: {e}")
        return

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        print(f"Nie można otworzyć pliku wideo: {args.video}")
        return

    metrics_data = []
    frame_count = 0
    total_start_time = time.time()
    
    print("Rozpoczęto inferencję. Naciśnij 'q' w oknie wideo, aby przerwać.")

    try:
        while cap.isOpened():
            success, frame = cap.read()
            if not success:
                break

            t_start = time.time()

            results = model.predict(
                source=frame, 
                imgsz=args.imgsz,
                conf=0.15, 
                verbose=False, 
                classes=[2, 3, 5, 7] 
            )

            t_end = time.time()
            real_fps = 1.0 / (t_end - t_start) if (t_end - t_start) > 0 else 0.0

            # --- NOWE: Wyciąganie liczby obiektów i ich pewności (confidence) ---
            boxes = results[0].boxes
            num_detections = len(boxes)
            
            if num_detections > 0:
                # Zamiana tensora z wartościami confidence na tablicę numpy, by wyciągnąć średnią
                conf_values = boxes.conf.cpu().numpy() if hasattr(boxes.conf, 'cpu') else boxes.conf
                avg_confidence = float(np.mean(conf_values))
            else:
                avg_confidence = 0.0
            # ----------------------------------------------------------------------

            speed_metrics = results[0].speed
            preprocess_ms = speed_metrics.get('preprocess', 0.0)
            inference_ms = speed_metrics.get('inference', 0.0)
            postprocess_ms = speed_metrics.get('postprocess', 0.0)
            tpu_total_ms = preprocess_ms + inference_ms + postprocess_ms

            metrics_data.append({
                'frame': frame_count,
                'img_size': args.imgsz,
                'preprocess_ms': round(preprocess_ms, 2),
                'inference_ms': round(inference_ms, 2),
                'postprocess_ms': round(postprocess_ms, 2),
                'total_latency_ms': round(tpu_total_ms, 2),
                'real_fps': round(real_fps, 2),
                'num_detections': num_detections,           # Dodano do metryk
                'avg_confidence': round(avg_confidence, 3)  # Dodano do metryk
            })

            if args.show:
                annotated_frame = results[0].plot()
                cv2.putText(annotated_frame, f"TPU Infer: {inference_ms:.1f}ms", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"FPS: {real_fps:.1f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(annotated_frame, f"Obiekty: {num_detections}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                cv2.putText(annotated_frame, f"Srednie Conf: {avg_confidence:.2f}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 165, 0), 2)
                
                cv2.imshow(f"YOLOv8 Edge TPU - {args.imgsz}", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("Przerwano ręcznie.")
                    break

            frame_count += 1
            if frame_count % 100 == 0:
                print(f"Przetworzono {frame_count} klatek. Obiekty: {num_detections}, Srednie Conf: {avg_confidence:.2f}, FPS: {real_fps:.2f}")

    except KeyboardInterrupt:
        print("Przerwano z klawiatury (Ctrl+C).")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        total_time = time.time() - total_start_time
        
        if metrics_data:
            file_exists = os.path.isfile(args.output_csv)
            with open(args.output_csv, mode='a', newline='') as csvfile:
                # Zaktualizowano nagłówki o nowe kolumny
                fieldnames = ['frame', 'img_size', 'preprocess_ms', 'inference_ms', 'postprocess_ms', 'total_latency_ms', 'real_fps', 'num_detections', 'avg_confidence']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                if not file_exists:
                    writer.writeheader()
                
                for row in metrics_data:
                    writer.writerow(row)
            
            # Obliczanie uśrednionych statystyk dla całego przebiegu
            avg_infer = sum(row['inference_ms'] for row in metrics_data) / len(metrics_data)
            avg_fps = sum(row['real_fps'] for row in metrics_data) / len(metrics_data)
            
            # Omijamy klatki bez detekcji przy liczeniu średniego confidence dla całego wideo
            frames_with_detections = [row['avg_confidence'] for row in metrics_data if row['num_detections'] > 0]
            global_avg_conf = sum(frames_with_detections) / len(frames_with_detections) if frames_with_detections else 0.0
            
            total_detections = sum(row['num_detections'] for row in metrics_data)
            avg_detections = total_detections / len(metrics_data)

            print(f"\n--- PODSUMOWANIE ({args.imgsz}x{args.imgsz}) ---")
            print(f"Całkowity czas: {total_time:.2f} s")
            print(f"Przetworzone klatki: {frame_count}")
            print(f"Średni czas inferencji TPU: {avg_infer:.2f} ms")
            print(f"Średnie FPS: {avg_fps:.2f}")
            print(f"Sumaryczna liczba detekcji: {total_detections}")
            print(f"Średnia liczba obiektów na klatkę: {avg_detections:.2f}")
            print(f"Średnia pewność (Confidence): {global_avg_conf:.3f}")
            print(f"Metryki zapisano do pliku: {args.output_csv}")
            print("-----------------------------------")

if __name__ == "__main__":
    main()