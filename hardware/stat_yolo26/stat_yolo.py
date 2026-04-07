import cv2
import time
from ultralytics import YOLO

log_filename = "performance_log.txt"
log_file = open(log_filename, "w")
log_file.write("Klatka | Preprocess (ms) | Inferencja (ms) | Postprocess (ms) | FPS\n")
log_file.write("-" * 75 + "\n")

model = YOLO('240_yolov8n_full_integer_quant_edgetpu.tflite')

video_path = "traffic.mp4"
cap = cv2.VideoCapture(video_path)

frame_count = 0
total_inference_time = 0

while cap.isOpened():
    success, frame = cap.read()
    
    if success:
        frame_count += 1
        
        start_time = time.time()
        
        results = model.predict(source=frame, show=False)
        
        fps = 1.0 / (time.time() - start_time)
        
        speeds = results[0].speed
        preprocess = speeds.get('preprocess', 0.0)
        inference = speeds.get('inference', 0.0)
        postprocess = speeds.get('postprocess', 0.0)
        
        total_inference_time += inference
        
        log_file.write(f"{frame_count:6d} | {preprocess:15.2f} | {inference:15.2f} | {postprocess:16.2f} | {fps:5.2f}\n")
        
        annotated_frame = results[0].plot()
        
        cv2.imshow("YOLO na Raspberry Pi + Coral", annotated_frame)
        
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        break

if frame_count > 0:
    avg_inference = total_inference_time / frame_count
    log_file.write("\n" + "=" * 75 + "\n")
    log_file.write("--- PODSUMOWANIE ---\n")
    log_file.write(f"Przetworzone klatki: {frame_count}\n")
    log_file.write(f"Średni czas inferencji: {avg_inference:.2f} ms\n")

log_file.close()
cap.release()
cv2.destroyAllWindows()

print(f"Zakończono! Logi wydajności zostały zapisane w pliku: {log_filename}")
