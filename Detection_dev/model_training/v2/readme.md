### YOLO26n car classification 

This version implements ongoing attempts of vehicle type detection and testing output by tracking using YOLO26n.
This version needs to be run in Google Colab. Includes automated publicly available dataset downloading from Roboflow.
https://universe.roboflow.com/archie-junio-dxv5t/car-detection-model-bwjpb
License: Public Domain
A car detection dataset sourced from the roads of EDSA in the philippines. 

### Status: 
Work in Progress. The current model not yet fully optimized. 
To do in future is retraining the model on a specialized dataset better suited for **CARLA** simulator environment and improving detection accuracy.

### How to Run
- Upload the `car_type_classification.ipynb` to Google Colab.
- Mount your Google Drive.
- Set your API_KEY. 
- Upload your test video to the Colab environment.
- Run the rest of cells, model will be saved in your Drive folder, access output video from Colab environment. 
