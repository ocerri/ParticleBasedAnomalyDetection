#!/usr/bin/env python
import os, sys, subprocess, re
from glob import glob
import argparse
import subprocess
import time, datetime
import numpy as np

inpath = '/eos/project/d/dshep/TOPCLASS/BSMAnomaly_IsoLep_lt_45_pt_gt_23/'
outpath = '/afs/cern.ch/user/o/ocerri/cernbox/ParticleBasedAnomalyDetection/data/'

SM_labels = ['Zll_lepFilter_13TeV', 'ttbar_lepFilter_13TeV', 'Wlnu_lepFilter_13TeV'] #, 'qcd_lepFilter_13TeV']
BSM_labels = ['leptoquark_LOWMASS_lepFilter_13TeV', 'hToTauTau_LOWMASS', 'hChToTauNu_LOWMASS', 'Ato4l_lepFilter_13TeV']

#____________________________________________________________________________________________________________
### processing the external os commands
def processCmd(cmd, quite = 0):
    status, output = subprocess.getstatusoutput(cmd)
    if (status !=0 and not quite):
        print('Error in processing command:\n   ['+cmd+']')
        print('Output:\n   ['+output+'] \n')
    return output


#_____________________________________________________________________________________________________________
#example line: python scripts/submitCondorJobs.py
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument ('--process', default=['SM+BSM'], type=str, help='Name of the sample', nargs='+')
    parser.add_argument ('--nFiles', help='number of files per jobs', default=500, type=int)
    parser.add_argument ('--maxFiles', help='Max files per process', default=-1, type=int)
    parser.add_argument ('--maxtime', help='Max wall run time [s=seconds, m=minutes, h=hours, d=days]', default='8h')

    parser.add_argument('-n', '--MaxPart', type=int, default=200, help='Max number of particles per event')
    parser.add_argument('--order', type=str, default='Pt', help='Particle variable used to order particles')

    args = parser.parse_args()
    process = []
    if len(args.process) == 1:
        if 'SM' in args.process[0].replace('BSM', ''):
            process += SM_labels
        if 'BSM' in args.process[0]:
            process += BSM_labels
        if len(process) == 0:
            process = args.process
    else:
        process = args.process

    time_scale = {'s':1, 'm':60, 'h':60*60, 'd':60*60*24}
    maxRunTime = int(args.maxtime[:-1]) * time_scale[args.maxtime[-1]]

    #Output directory
    date = datetime.date.today()
    outdir = '{}{:02d}{:02d}_{}part_{}Order_v3/'.format(date.year, date.month, date.day, args.MaxPart, args.order)
    outdir = outpath + outdir
    if os.path.isdir(outdir):
        print('Folder '+ outdir +' already existing')
        # exit()
    else:
        print('Creating the directory structure')
        os.system('mkdir -p ' + outdir)
        os.system('mkdir -p ' + outdir+'/out')
        os.system('mkdir -p ' + outdir+'/cfg')

    for p in process:
        if args.maxFiles <= 0:
            N_files = len(glob(inpath + p +'/*.h5'))
        else:
            N_files = len(glob(inpath + p +'/*.h5')[:args.maxFiles])
        print('Process', p, '({})'.format(N_files))
        start_file = np.arange(0, N_files, args.nFiles)

        for stf in start_file:
            subname = p+'_'+str(stf)+'.sub'
            fsub = open(subname, 'w')
            fsub.write('executable    = /afs/cern.ch/user/o/ocerri/cernbox/ParticleBasedAnomalyDetection/scripts/condorJob.sh')
            fsub.write('\n')
            exec_args = p + ' {} {} {} {}'.format(args.MaxPart, args.order, stf, args.nFiles)
            fsub.write('arguments     = ' + exec_args)
            fsub.write('\n')
            fsub.write('output        = {}/out/{}_{}.$(ClusterId).$(ProcId).out'.format(outdir, p, str(stf)))
            fsub.write('\n')
            fsub.write('error         = {}/out/{}_{}.$(ClusterId).$(ProcId).err'.format(outdir, p, str(stf)))
            fsub.write('\n')
            fsub.write('log           = {}/out/{}_{}.$(ClusterId).$(ProcId).log'.format(outdir, p, str(stf)))
            fsub.write('\n')
            fsub.write('+MaxRuntime   = '+str(maxRunTime))
            fsub.write('\n')
            fsub.write('queue 1')
            fsub.write('\n')
            fsub.close()

            print('Submitting job ' + subname)
            output = processCmd('condor_submit ' + subname)
            print('Job submitted')
            os.rename(subname, outdir+'/cfg/'+subname)
