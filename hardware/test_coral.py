from ultralytics import YOLO
import time

model = YOLO('yolo26n_full_integer_quant.tflite', task='detect')  

source_img = 'https://ultralytics.com/images/bus.jpg'

print("Rozpoczynam detekcję na YOLOv26n...")
start_time = time.time()

results = model.predict(source=source_img, save=True, conf=0.3, stream=False)

end_time = time.time()
print(f"Pełny czas operacji (ładowanie + detekcja): {round(end_time - start_time, 3)}s")

for result in results:
    boxes = result.boxes
    for box in boxes:
        class_name = model.names[int(box.cls[0])]
        confidence = float(box.conf[0])
        print(f"--> Wykryto: {class_name} ({round(confidence * 100, 1)}%)")

print("\nGotowe! Sprawdź folder 'runs/detect', żeby zobaczyć wynikowy plik.")
