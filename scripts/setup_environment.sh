MINI_DIR=$(pwd)/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -p $MINI_DIR

source $MINI_DIR/etc/profile.d/conda.sh
conda init bash

exec bash

conda env create -f environment.yml
conda activate mamba-amt