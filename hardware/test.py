from ultralytics import YOLO

# Load a model
model = YOLO("./yolo26n_full_integer_quant.tflite")  # Load an official model or custom model

# Run Prediction
model.predict("./sample.jpg")  # Inference defaults to the first TPU

model.predict("./sample.jpg", device="tpu:0")  # Select the first TPU

