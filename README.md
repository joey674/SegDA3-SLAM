# deploy
sudo apt-get install git python3-pip libboost-all-dev cmake gcc g++ unzip

conda create -n vggt-slam python=3.11
conda activate vggt-slam

## SLAM
pip3 install -r requirements.txt
git clone https://github.com/Dominic101/salad.git
pip install -e ./salad

pip install -e .

## DA3
cd /home/zhouyi/repo/model_DepthAnythingV3
pip install xformers torch\>=2 torchvision
pip install -e . 
cd /home/zhouyi/repo/VGGT-SLAM



# run
python3 main.py --image_folder office_loop --max_loops 1 --vis_map