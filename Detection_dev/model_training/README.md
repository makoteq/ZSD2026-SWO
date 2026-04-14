# VEHICLE DETECTION AND CLASSIFICATION PIPELINE

## WHAT DOES MAIN.PY

Detects and tracks vehicles in a video using a custom YOLO model (`best.pt`),  
classifies each vehicle type (coupe, hatchback, sedan, suv, truck, van) using a CNN model,  
and draws bounding boxes, labels and trajectories on the output video.

Each detected vehicle is stored as a **Car object** with:
- **type**: vehicle category from CNN  
- **k**: CNN-predicted deceleration coefficient (k = 1 / (2 * mu * g))  
- **breakingDistance**: estimated braking distance (d = v^2 * k), updated every frame  
- **speed**: placeholder, **NOT YET IMPLEMENTED** - **history**: list of (x, y) positions across frames, used to draw trajectory  
- **lastCrop**: best crop of the vehicle (taken at highest YOLO confidence)  
- **maxConfidence**: highest YOLO detection confidence seen for this vehicle  

CNN is only called when YOLO confidence exceeds the previous best for that vehicle,  
so the classification is always based on the clearest view.

---

## REQUIREMENTS

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

## STRUCTURE

```text
../  
    main.py                            
    content/
        best.pt                                 <- custom YOLO detection model
        carla_classification_cnn_model_v01.h5   <- CNN classification model
        traffic.mp4                             <- input video to process
```

The `content/` folder must be created manually and the model files and video placed inside.

The script will:
1. Load YOLO and CNN models from `content/`
2. Open the input video from `content/traffic.mp4`
3. Process every frame - detect, track and classify vehicles
4. Write the annotated output video to `content/trajectory.mp4`
5. Save a notebook with vehicle crops to `content/cars_output.ipynb`, optional

---

## OUTPUT

Both files appear in `content/` after the script finishes:

### trajectory.mp4
Annotated video with bounding boxes, vehicle type, track ID and trajectory lines.

### cars_output.ipynb
Jupyter notebook - one cell per detected vehicle showing its crop image, type, YOLO confidence and k value. Open in Jupyter or VSCode to view. This can be removed from the script if not needed (see save_notebook call at end of main).

---

## CONFIGURATION

All parameters are at the top of `tracking.py` under the CONFIGURATION section:

```text
  VIDEO_PATH          - path to input video
  START_TIME          - start time in seconds (skips the beginning of the video)
  CONF_THRESHOLD      - minimum YOLO detection confidence (default 0.5)
  IMGSZ               - YOLO input image size, 416 or 800 depending on best.pt
  MAX_MISSING_FRAMES  - how many frames a vehicle can be unseen before being removed
                        set to float('inf') to never remove vehicles from tracking
  TRACK_COLOR         - trajectory line color (BGR)
  BBOX_COLOR          - bounding box color (BGR)
```

---

## KNOWN LIMITATIONS / TODO

- speed is not implemented yet, breakingDistance will always be 0.0 until then
- CNN is trained on CARLA synthetic data, real-world accuracy may vary
- k value is taken from the best confidence frame only, not averaged over time