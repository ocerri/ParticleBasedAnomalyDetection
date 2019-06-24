import h5py
from glob import glob
import numpy as np
import os
import argparse
import datetime

inpath = '/eos/project/d/dshep/TOPCLASS/BSMAnomaly_IsoLep_lt_45_pt_gt_23/'
outpath = '/afs/cern.ch/user/o/ocerri/cernbox/ParticleBasedAnomalyDetection/data/'

SM_labels = ['qcd_lepFilter_13TeV', 'ttbar_lepFilter_13TeV', 'Wlnu_lepFilter_13TeV', 'Zll_lepFilter_13TeV']
BSM_labels = ['leptoquark_LOWMASS_lepFilter_13TeV', 'Ato4l_lepFilter_13TeV']


parser = argparse.ArgumentParser()
parser.add_argument('sample_label', type=str, help='Name of the sample', nargs='+')
parser.add_argument('-N', '--MaxEvts', type=int, default=1000, help='Max number of events')
parser.add_argument('-n', '--MaxPart', type=int, default=10, help='Max number of particles per event')
parser.add_argument('--order', type=str, default='Pt', help='Particle variable used to order particles')
parser.add_argument('-i', '--input_path', type=str, default=inpath)
parser.add_argument('-o', '--output_path', type=str, default=outpath)
parser.add_argument('-F', '--force', action='store_true', default=False)
args = parser.parse_args()

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
outdir = '{}{:02d}{}_{}part_{}Order/'.format(date.year, date.month, date.day, args.MaxPart, args.order)
outdir = args.output_path + outdir

if os.path.isdir(outdir):
    if args.force:
        os.system('rm -rf ' + outdir)
        os.system('mkdir -p ' + outdir)
    else:
        print('Folder already existing')
        exit()
else:
    os.system('mkdir -p ' + outdir)


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
        if i%100 == 0 or i == len(file_list)-1:
            print('At file', i, 'size:', dataset.shape[0], 'errors:', errors)
        try:
            f = h5py.File(fname, 'r')

            # print(f['Particles_Names'][()].astype(np.str))
            if np.max(f['Particles_Names'][()].astype(np.str) == args.order) == False:
                print('Ordeing variable ({}) not found in file {}'.format(args.order, fname))
                errors += 1
                continue

            i_v = np.argmax(f['Particles_Names'][()].astype(np.str) == args.order)

            parts_pt = f['Particles'][()][:,:,i_v]
            idx = np.argpartition(parts_pt, -args.MaxPart)[:,-args.MaxPart:] #get the indexes of the top MaxPart ordered by i_v

            for particles, sel in zip(f['Particles'][()], idx):
                parts = particles[sel]

                pt_eta_phi = parts[:,5:8]
                charge = parts[:, -1]
                pId = 0*parts[:, 16] + 1*parts[:, 17] + 2*parts[:, 18] + 3*parts[:, 14] + 4*parts[:, 15]
                features = np.column_stack((pt_eta_phi, charge, pId)).astype(np.float16).reshape((1, args.MaxPart, 5))

                dataset = np.concatenate((dataset, features))

                if args.MaxEvts > 0 and dataset.shape[0] >= args.MaxEvts:
                    break
        except:
            errors += 1
            print('[{}]'.format(errors), fname, 'failed')

        if args.MaxEvts > 0 and dataset.shape[0] >= args.MaxEvts:
            break

    print('Saving dataset with {} entries: '.format(dataset.shape[0]), outname)
    np.save(outname, dataset)
    print('\n')

#
#         #Remove Sphericity
#         if hlf.shape[1] == 24 and list(f['HLF_Names'])[5] == 'SPH':
#             hlf = np.delete(hlf, 4, 1)
#         elif hlf.shape[1] != 23:
#             print('Non matching shapes ---> Exiting')
#             continue
#
#         #Change from radial to cathesian MET
#         METp = hlf[:,1]*np.cos(hlf[:,2])
#         METo = hlf[:,1]*np.sin(hlf[:,2])
#         hlf[:,1] = METp
#         hlf[:,2] = METo
#
#         hlf_train = np.concatenate((hlf_train, hlf))
#
#     if hlf_train.shape[0] > args.MaxNumber:
#         print('Max number of {} overcome'.format(args.MaxNumber))
#         print('At file', i, 'size:', hlf_train.shape[0], 'errors:', errors)
#         break
#
#
# print(hlf_train.shape)
#
# if hlf_train.shape[0] > 0:
#     np.save(outname, hlf_train)
# else:
#     print('File empty. No output.')
