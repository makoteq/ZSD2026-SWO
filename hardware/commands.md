yolo export model=path/to/model.pt format=edgetpu # Export an official model or custom model

yolo predict model=path/to/edgetpu_model.tflite source=path/to/source.png # Load an official model or custom model

echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install libedgetpu1-std
