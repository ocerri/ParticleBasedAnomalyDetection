if ! [ -x "$(command -v conda)" ]; then
  echo 'Install conda first (https://docs.conda.io/en/latest/miniconda.html)' >&2
  exit 1
fi

conda create -n PartAN python=3
conda activate PartAN

conda install h5py numpy=1.16.4
conda install jupyter

conda install pytorch torchvision -c pytorch
# With cuda supports:
# MacOS Binaries dont support CUDA, install from source if CUDA is needed
# conda install pytorch torchvision cudatoolkit=10.0 -c pytorch

conda install matplotlib
conda install -c conda-forge prettytable
pip install gpustat

# Not used for the moment
# pip install tensorflow==2.0.0-beta1
# conda install -c conda-forge root

# Optional
# conda install -c conda-forge jupyter_contrib_nbextensions
