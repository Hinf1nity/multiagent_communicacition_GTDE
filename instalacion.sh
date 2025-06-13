chmod +x install_sc2.sh
chmod +x project/scripts/train_smacv2/protoss_20v20.sh
apt-get update
apt-get install unzip wget -y
pip install gym
pip install setproctitle
pip install tensorboardX
pip install wandb
pip install torch_geometric
pip install git+https://github.com/oxwhirl/smacv2.git
