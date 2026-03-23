from ultralytics import YOLO

# Load a model
model = YOLO("./240_yolov8n_full_integer_quant_edgetpu.tflite")  # Load an official model or custom model


model.predict("./sample.jpg", device="tpu:0")  # Select the first TPU

