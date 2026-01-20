# SegDA3

## deploy
```bash
sudo apt-get install git python3-pip libboost-all-dev cmake gcc g++ 
conda create -n (vggt-slam)da3-slam python=3.11
conda activate (vggt-slam)da3-slam
```
### SLAM
```bash
pip3 install -r requirements.txt
git clone https://github.com/Dominic101/salad.git
pip install -e ./salad
pip install -e .
### DA3
```bash
cd /home/zhouyi/repo/model_DepthAnythingV3
pip install xformers torch\>=2 torchvision
pip install -e . 
cd /home/zhouyi/repo/VGGT-SLAM
```
## run
```bash
python3 main.py --image_folder office_loop
python3 main.py --image_folder /home/zhouyi/repo/dataset/2077/scene1 

python src/seg_da3/SegDA3_eval.py 
```