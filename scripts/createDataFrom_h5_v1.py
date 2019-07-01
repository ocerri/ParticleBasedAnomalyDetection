import h5py
from glob import glob
import numpy as np
import os
import argparse
import datetime
import time

inpath = '/eos/project/d/dshep/TOPCLASS/BSMAnomaly_IsoLep_lt_45_pt_gt_23/'
outpath = '/afs/cern.ch/user/o/ocerri/cernbox/ParticleBasedAnomalyDetection/data/'

SM_labels = ['Zll_lepFilter_13TeV', 'ttbar_lepFilter_13TeV', 'Wlnu_lepFilter_13TeV', 'qcd_lepFilter_13TeV']
BSM_labels = ['leptoquark_LOWMASS_lepFilter_13TeV', 'Ato4l_lepFilter_13TeV']


parser = argparse.ArgumentParser()
parser.add_argument('sample_label', type=str, help='Name of the sample', nargs='+')
parser.add_argument('-N', '--MaxEvts', type=int, default=1000, help='Max number of events')
parser.add_argument('-n', '--MaxPart', type=int, default=20, help='Max number of particles per event')
parser.add_argument('--order', type=str, default='Pt', help='Particle variable used to order particles')
parser.add_argument('-i', '--input_path', type=str, default=inpath)
parser.add_argument('-o', '--output_path', type=str, default=outpath)
parser.add_argument('-F', '--force', action='store_true', default=False)
args = parser.parse_args()

if args.MaxPart < 15:
    print('Number of particles per event too low, resetting it to 15.')
    args.MaxPart = 15

if len(args.sample_label) == 1:
    if args.sample_label[0] == 'SM':
        args.sample_label = SM_labels
    if args.sample_label[0] == 'BSM':
        args.sample_label = BSM_labels

print('Running on:')
print(args.sample_label)
print('')

#Output directory
date = datetime.date.today()
outdir = '{}{:02d}{}_{}part_{}Order_v1/'.format(date.year, date.month, date.day, args.MaxPart, args.order)
outdir = args.output_path + outdir

if os.path.isdir(outdir):
    if args.force:
        os.system('rm -rf ' + outdir)
        os.system('mkdir -p ' + outdir)
    else:
        print('Folder already existing')
        if 'y' != input('Continue? [y/n]\n'):
            print('Exiting...')
            exit()
        else: print('')
else:
    os.system('mkdir -p ' + outdir)


last_time_printed = time.time()
for sample_label in args.sample_label:
    outname = outdir+sample_label+'.npy'

    if os.path.isfile(outname):
        if args.force:
            os.remove(outname)
        else:
            print('File '+outname+' already existing')
            continue

    dataset = np.zeros((0, args.MaxPart, 5)).astype(np.float16)

    file_list = glob(args.input_path + sample_label +'/*.h5')
    print(sample_label, '({} files)'.format(len(file_list)))
    errors = 0
    for i, fname in enumerate(file_list):
        if errors > 10:
            print('Too many errors')
            exit(0)
        if time.time() - last_time_printed > 30. or i%100 == 0 or i == len(file_list)-1:
            print('At file', i, 'size:', dataset.shape[0], 'errors:', errors)
            last_time_printed = time.time()
        try:
            f = h5py.File(fname, 'r')

            # print(f['Particles_Names'][()].astype(np.str))
            if np.max(f['Particles_Names'][()].astype(np.str) == args.order) == False:
                print('Ordeing variable ({}) not found in file {}'.format(args.order, fname))
                errors += 1
                continue

            i_v = np.argmax(f['Particles_Names'][()].astype(np.str) == args.order)

            for particles in f['Particles'][()]:

                # Get the muons
                sel = particles[:, 18] > 0.5
                # if np.sum(sel) < 5: continue
                muons = particles[sel]
                if muons.shape[0] > 5:
                    muons_v = muons[:,i_v]
                    idx = np.argpartition(muons_v, -5)[-5:]
                    muons = muons[idx]

                # Get the electrons
                sel = particles[:, 17] > 0.5
                electrons = particles[sel]
                Nmax_ele = 10 - muons.shape[0]
                if electrons.shape[0] > Nmax_ele:
                    ele_v = electrons[:,i_v]
                    idx = np.argpartition(ele_v, -Nmax_ele)[-Nmax_ele:]
                    electrons = electrons[idx]

                # Get photons and hadrons
                sel = particles[:, 14] + particles[:, 15] + particles[:, 16] > 0.5
                parts = particles[sel]
                Nmax_parts = args.MaxPart - muons.shape[0] - electrons.shape[0]
                if parts.shape[0] > Nmax_parts:
                    parts_v = parts[:,i_v]
                    idx = np.argpartition(parts_v, -Nmax_parts)[-Nmax_parts:]
                    parts = parts[idx]

                parts = np.concatenate((muons, electrons, parts))
                pt_eta_phi = parts[:,5:8]
                #Rescale Pt
                parts[:,5] /= 10.
                charge = parts[:, -1]
                # 0 = Muon, 1 = Electron, 2 = Photon, 3 = Charged hadron, 4 = Neutral hadron
                pId = 0*parts[:, 18] + 1*parts[:, 17] + 2*parts[:, 16] + 3*parts[:, 14] + 4*parts[:, 15]

                features = np.column_stack((pt_eta_phi, charge, pId)).astype(np.float16).reshape((1, args.MaxPart, 5))
                dataset = np.concatenate((dataset, features))

                if args.MaxEvts > 0 and dataset.shape[0] >= args.MaxEvts:
                    break
        except:
            errors += 1
            print('[{}]'.format(errors), fname, 'failed')

        if args.MaxEvts > 0 and dataset.shape[0] >= args.MaxEvts:
            break

    print('Saving dataset with {} entries from {} files: '.format(dataset.shape[0], i+1), outname)
    np.save(outname, dataset)
    print('\n')
