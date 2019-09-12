import numpy as np
import argparse
from glob import glob
import os
# from progressBar import ProgressBar

parser = argparse.ArgumentParser()
parser.add_argument('input', type=str, nargs='+')
parser.add_argument('-o', '--output', type=str)
parser.add_argument('-r', '--remove', action='store_true', default=False)
args = parser.parse_args()

if len(args.input) == 1:
    args.input = glob(os.environ['PWD']+'/'+args.input[0])
flist = args.input

print('Merging {} files...'.format(len(flist)))
# pb = ProgressBar(1+len(flist))
# pb.show(0)
for i, fname in enumerate(flist):
    # pb.show(i+1)
    print(i)
    if i == 0:
        dataset = np.load(fname)
    else:
        dataset = np.concatenate((dataset, np.load(fname)))

print('Saving dataset with {} entries from {} files: '.format(dataset.shape[0], i+1), args.output)
np.save(args.output, dataset)

if args.remove:
    print('Cleaning files...')
    for fname in flist:
        os.system('rm ' + fname)
