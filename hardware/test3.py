import cv2
from ultralytics import YOLO


model = YOLO('240_yolov8n_full_integer_quant_edgetpu.tflite')


video_path = "traffic.mp4"
cap = cv2.VideoCapture(video_path)

while cap.isOpened():
    success, frame = cap.read()
    
    if success:
    
        results = model.predict(source=frame, show=False)
        
    
        annotated_frame = results[0].plot()
        
   
        cv2.imshow("YOLOv8 na Raspberry Pi + Coral", annotated_frame)
        
    
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:

        break

cap.release()
cv2.destroyAllWindows()