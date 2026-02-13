# DynaDA3

## deploy
```bash
### conda
sudo apt-get update
sudo apt-get install git python3-pip libboost-all-dev cmake gcc g++ 
conda create -n dynada3-slam python=3.11
conda activate dynada3-slam
### SLAM
pip3 install -r requirements.txt
git clone https://github.com/Dominic101/salad.git
pip install -e ./salad
pip install -e .
### DA3
pip install xformers torch\>=2 torchvision
pip install -e ../DynaDA3
```

## run
```bash
python3 main.py  
python3 main.py --image_folder ../dataset/wildgs-slam/wildgs_racket_test
python3 main.py --image_folder ../dataset/UKA/UKA_Case1Part1Scene1

```