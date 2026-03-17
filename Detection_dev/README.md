# YOLOv7 Traffic Analysis (software branch)

Detects vehicles in video streams within an ROI polygon using YOLOv7. Processes video frame-by-frame, identifies cars/trucks/buses/motorcycles, and marks objects passing through the defined region.

## Setup

```bash
# 1. Clone YOLOv7
git clone https://github.com/WongKinYiu/yolov7.git

# 2. Install dependencies
pip install -r requirements.txt

# 3. Download model
wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
```

## What It Does

- Loads YOLOv7 pretrained weights
- Reads video file and processes frames within defined time range
- Detects vehicles (classes: car, truck, bus, motorcycle)
- Highlights detections inside ROI polygon with red bboxes
- Displays live output with visualization