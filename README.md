# deploy
sudo apt-get install git python3-pip libboost-all-dev cmake gcc g++ unzip

conda create -n vggt-slam python=3.11
conda activate vggt-slam
./setup.sh

# run
python3 main.py --image_folder office_loop --max_loops 1 --vis_map