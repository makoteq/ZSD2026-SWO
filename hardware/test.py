from ultralytics import YOLO

# Load a model
model = YOLO("./yolo26n_full_integer_quant.tflite")  # Load an official model or custom model

# Run Prediction
model.predict("./sample.jpg")
