# SegDA3

## deploy
```bash
### conda
sudo apt-get install git python3-pip libboost-all-dev cmake gcc g++ 
conda create -n segda3-slam python=3.11
conda activate segda3-slam
### SLAM
pip3 install -r requirements.txt
git clone https://github.com/Dominic101/salad.git
pip install -e ./salad
pip install -e .
### DA3
cd /home/zhouyi/repo/SegDA3
pip install xformers torch\>=2 torchvision
pip install -e . 
cd /home/zhouyi/repo/SegDA3_SLAM
```

## run
```bash
python3 main.py --image_folder 
python3 main.py --image_folder /home/zhouyi/repo/dataset/2077/scene1 
python3 main.py --image_folder /home/zhouyi/repo/dataset/wildgs-slam/wildgs_racket_test

```