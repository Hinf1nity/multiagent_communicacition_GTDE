sudo chmod +x install_sc2,sh
sudo chmod +x project/scripts/train_smacv2/protoss_20v20.sh
cd ~
sudo apt-get update
sudo apt-get install unzip wget -y
pip install wandb gym setproctitle socket tensorboardX
pip install git+https://github.com/oxwhirl/smacv2.git
