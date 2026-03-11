
Progres na tydzień pierwszy realizowany w ramach zadania konfiguracji środowiska do transfer-learningu i przygotowaniem datasetu. 

# Sebastian

1. Wybór database do wstępnego treningu modelu YOLO. 
Zbiór danych z platformy Roboflow sformatowany jest zgodnie ze standardem Ultralytics YOLO format.
https://universe.roboflow.com/archie-junio-dxv5t/car-detection-model-bwjpb

Provided by a Roboflow user
License: Public Domain

A car detection dataset sourced from the roads of EDSA in the philippines. 
The dataset includes 5112 images.
Cars are annotated in YOLO26 format.

2. Przygotowanie wstępnego skryptu train_from_roboflow.ipynb odpowiedzialnego za pobranie danych przez API Roboflow, transfer-learning instancji modelu YOLO26n. 
Zapisanie dotychczasowo najlepszych wag modelu runs/detect/yolo_car/weights/best.pt wymagających rozwoju oraz predykcja (tracking) na podstawie pliku wideo.

Środowisko treningowe: Google Colab (t4 GPU)

3. Trwają prace nad poprawą jakości detekcji pojazdów aby umożliwić dalszą pracę nad estymacją drogi hamowania pojazdu.



