# Hardware (software branch)
This branch contains the necessary scripts and configuration files required to deploy the trained object detection model onto the target hardware platform.


## Prerequisites
Before starting the installation, ensure that your environment meets the following software and hardware requirements.
```bash
Python 3.6 to 3.9

Google Coral on USB 3.0 port
```
## Exporting model
To run the model on the Google Coral TPU, you must first convert the standard PyTorch weights (.pt) into a compatible Edge TPU format. This process should be performed on a Linux-based system or via WSL for compatibility with compiler tools.
```bash
yolo export model=path/to/model.pt format=edgetpu # Export an official model or custom model
```

## Setup
Follow these steps to prepare the Raspberry Pi environment and install the required runtimes for hardware acceleration:
```bash
# Update OS
sudo apt-get upgrade
sudo apt-get upgrade

# Install or upgrade the ultralytics package from PyPI
pip install -U ultralytics[export]

# Installing the Edge TPU runtime
sudo dpkg -i libedgetpu1-std_16.0tf2.19.1-1.bookworm_arm64.deb

# Install dependencies
pip install tensorflow 
pip install -U tflite-runtime
pip install torch==2.3.1
pip install torchvision==0.18.1


```
## What It Does
This module is responsible for the final execution of the safety system on the embedded platform. It ensures stable operation on the Raspberry Pi by monitoring system vitals, memory and processing power, while managing hardware-accelerated inference on the Edge TPU.

* **Hardware Acceleration**: Offloads complex neural network computations from the Raspberry Pi CPU to the dedicated Edge TPU.
* **Low-Latency Inference**: Enables real-time processing of high-frequency data streams for immediate response.
* **Efficiency Optimization**: Significantly reduces power consumption and heat generation during continuous object detection tasks.
* **Performance Monitoring**: Tracks system vitals, such as memory overhead and processing latency.
