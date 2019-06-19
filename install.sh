if ! [ -x "$(command -v conda)" ]; then
  echo 'Install conda first (https://docs.conda.io/en/latest/miniconda.html)' >&2
  exit 1
fi

conda create -n PartAN python=3
conda activate PartAN

conda install h5py numpy
pip install tensorflow==2.0.0-beta1
conda install jupyter
# Not used for the moment
# conda install -c conda-forge root
