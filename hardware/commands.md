yolo export model=path/to/model.pt format=edgetpu # Export an official model or custom model

yolo predict model=yolo26n_full_integer_quant_edgetpu.tflite source=sample.jpg # Load an official model or custom model
