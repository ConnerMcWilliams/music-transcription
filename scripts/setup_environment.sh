MINI_DIR=$(pwd)../miniconda3

if test -f "$MINI_DIR"; then
    echo "$MINI_DIR exists"
else 
    echo "$MINI_DIR does not exist. Downloading..."    
    wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
fi
bash Miniconda3-latest-Linux-x86_64.sh -b -p $MINI_DIR

source $MINI_DIR/etc/profile.d/conda.sh
conda init bash

exec bash

conda env create -f environment.yml
conda activate mamba-amt