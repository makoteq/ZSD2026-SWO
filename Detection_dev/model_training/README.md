# VEHICLE DETECTION AND CLASSIFICATION PIPELINE

## MAIN.PY

Detects and tracks vehicles in the input video from `content/traffic.mp4` video using a custom YOLO model (`best.pt`),  
classifies each vehicle type (coupe, hatchback, sedan, suv, truck, van) using a CNN model,  
and draws bounding boxes, labels and trajectories on the output video.  
Writes the annotated output video to `content/trajectory.mp4`,  
Optionally Saves a notebook with vehicle crops to `content/cars_output.ipynb`.

Access to model:
https://drive.google.com/drive/folders/1JhdJ7mt9RcEYkpEYt98uyG6I3Pz76pQ5?usp=sharing


---

### REQUIREMENTS

Python **3.12** is required — `tensorflow 2.16.1` does not support newer versions.

Install dependencies:
```bash
pip install ultralytics==8.4.37
pip install torch==2.11.0
pip install torchvision==0.26.0
pip install tensorflow==2.16.1
pip install opencv-python==4.10.0.84
pip install numpy==1.26.4
pip install pillow==12.1.1
pip install nbformat
```

---

### STRUCTURE

```text
../  
    main.py                            
    content/
        best.pt                                 <- custom YOLO detection model
        carla_classification_cnn_model_v01.h5   <- CNN classification model
        traffic.mp4                             <- input video to process
```


---

## YOLO_training FOLDER

### main.ipynb

Script for testing a trained on a custom dataset YOLOv8 model on video data with live visualization of detections.

### training_carla.ibynb

Training script for YOLOv8 model on custom CARLA dataset.  
Uses Roboflow to access prepared and annotated data. Link to access dataset is given below:  
https://app.roboflow.com/mareks-workspace-entpb/carla-osobowki-przejscie/4


## vehicle_classification_CNN FOLDER

### dataset.py 

XXX

### main.ipynb

XXX

## metrics FOLDER

### metrics.ipynb

Compares two models and collects performance metrics,  runs tracking on multiple video segments and evaluates performance across different weather conditions.  
Shows for each model and segment: FPS, inference time, object detections per frame, confidences, number of captured cars during tracking. 

### precision_recall_evaluation.ipynb 

This script manually evaluates YOLOv8 models on a labeled dataset by comparing predictions with ground truth annotations.
It calculates Precision and Recall for a selected class using IoU-based matching.
Supports class mapping between dataset and model outputs for fair comparison between different models.

### video_compare.ipynb

This script compares a base YOLOv8 model with a custom-trained model. It runs both on a video, performs object detection, and saves annotated results.



