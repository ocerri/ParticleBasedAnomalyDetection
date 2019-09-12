#!/bin/bash
source /afs/cern.ch/user/o/ocerri/.bash_profile

cd /afs/cern.ch/user/o/ocerri/cernbox/ParticleBasedAnomalyDetection
source setup.sh

which python

python scripts/createDataFrom_h5_v3.py $1 -n $2 --order $3 --startFile $4 --nFiles $5 -F -N 1000000
