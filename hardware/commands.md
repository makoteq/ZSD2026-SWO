yolo export model=path/to/model.pt format=edgetpu # Export an official model or custom model

yolo predict model=path/to/edgetpu_model.tflite source=path/to/source.png # Load an official model or custom model

echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" | sudo tee /etc/apt/sources.list.d/coral-edgetpu.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install libedgetpu1-std



########################INSTALL PYTHON 3.9.12#########################
sudo rm /usr/lib/python3.11/EXTERNALLY-MANAGED
pip3 install --upgrade thonny


1)sudo apt-get update
2)sudo apt-get upgrade
3)curl https://pyenv.run | bash
4)echo 'export PATH="$HOME/.pyenv/bin:$PATH"' >> ~/.bashrc
echo 'eval "$(pyenv init --path)"' >> ~/.bashrc
echo 'eval "$(pyenv virtualenv-init -)"' >> ~/.bashrc
exec "$SHELL"

5)sudo apt-get install --yes libssl-dev zlib1g-dev libbz2-dev libreadline-dev libsqlite3-dev llvm libncurses5-dev libncursesw5-dev xz-utils tk-dev libgdbm-dev lzma lzma-dev tcl-dev libxml2-dev libxmlsec1-dev libffi-dev liblzma-dev wget curl make build-essential openssl

6)pyenv install 3.9.12

7)pyenv local 3.9.12

8)python --version



#########################Google Coral Usb######################################

1)python3 -m venv .venv
source .venv/bin/activate

2)pip install edge-tpu-silva
3)pip install numpy==1.26.4
####Restart Raspberry pi 4#########
4)silvatpu-linux-setup
####Restart Raspberry pi 4 ########
############################################################AFTER ALL THIS PROCESS  COMPLETED JUST REBOOT YOUR RASPBERRY PI 4##########################################################################
