if ! [ -x "$(command -v conda)" ]; then
  echo 'Install conda first (https://docs.conda.io/en/latest/miniconda.html)' >&2
  exit 1
fi

conda create -n PartAN python=3.7.3
conda activate PartAN

conda install h5py=2.9.0 numpy=1.16.4 jupyter matplotlib=3.1.0 scikit-learn=0.21.2
conda install -c conda-forge prettytable jupyter_contrib_nbextensions
pip install gpustat

# conda install pytorch=1.1.0 torchvision=0.3.0 -c pytorch
# With cuda supports:
# MacOS Binaries dont support CUDA, install from source if CUDA is needed
conda install pytorch=1.1.0 torchvision=0.3.0 cudatoolkit=10.0 -c pytorch

# Not used for the moment
# pip install tensorflow==2.0.0-beta1
# conda install -c conda-forge root
