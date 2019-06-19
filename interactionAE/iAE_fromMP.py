import setGPU
import os
import numpy as np
import h5py
import glob
import itertools
import sys
from sklearn.utils import shuffle

import torch
import torch.nn as nn
from torch.autograd.variable import *
import torch.optim as optim

from generatorIN import InEventLoader
import random

args_cuda = bool(sys.argv[2])

class GraphNet(nn.Module):
    def __init__(self, n_constituents, params, hidden, De, Do, verbose = False):
        super(GraphNet, self).__init__()
        self.hidden = hidden
        self.P = len(params)
        self.Nin = n_constituents
        # not general, but we take it simple here
        self.Nout = n_constituents
        self.Nr_in = self.Nin * (self.Nin - 1)
        self.Nr_out = self.Nin * (self.Nin - 1)
        self.De = De
        self.Do = Do
        self.verbose = verbose
        self.assign_matrices()

        # encoder IN layer: NxP -> NxDo -> Do (bottle neck size)
        self.fr1_enc = nn.Linear(2 * self.P, hidden).cuda()
        self.fr2_enc = nn.Linear(hidden, int(hidden/2)).cuda()
        self.fr3_enc = nn.Linear(int(hidden/2), self.De).cuda()
        self.fo1_enc = nn.Linear(self.P + self.De, hidden).cuda()
        self.fo2_enc = nn.Linear(hidden, int(hidden/2)).cuda()
        self.fo3_enc = nn.Linear(int(hidden/2), self.Do).cuda()

        # bottle neck
        self.ConvTranspose = nn.ConvTranspose1d(in_channel=1, out_channel=self.Nr_out, kernel_size=1, stride=1, padding=0)

        # decoder IN layer: from Do -> NxDo -> NxP
        self.fr1_dec = nn.Linear(2 * self.Do, hidden).cuda()
        self.fr2_dec = nn.Linear(hidden, int(hidden/2)).cuda()
        self.fr3_dec = nn.Linear(int(hidden/2), self.De).cuda()
        self.fo1_dec = nn.Linear(self.Do + self.De, hidden).cuda()
        self.fo2_dec = nn.Linear(hidden, int(hidden/2)).cuda()
        self.fo3_dec = nn.Linear(int(hidden/2), self.P).cuda()
        
        
    def assign_matrices(self):
        ### encoder IN layer
        self.Rr_enc = torch.zeros(self.Nin, self.Ninr)
        self.Rs_enc = torch.zeros(self.Nin, self.Ninr)
        receiver_sender_list = [i for i in itertools.product(range(self.Nin), range(self.Nin)) if i[0]!=i[1]]
        for i, (r, s) in enumerate(receiver_sender_list):
            self.Rr_enc[r, i] = 1
            self.Rs_enc[s, i] = 1
        self.Rr_enc = Variable(self.Rr_enc).cuda()
        self.Rs_enc = Variable(self.Rs_enc).cuda()
        ### decoder IN layer
        self.Rr_dec = torch.zeros(self.Nout, self.Noutr)
        self.Rs_dec = torch.zeros(self.Nout, self.Noutr)
        receiver_sender_list = [i for i in itertools.product(range(self.Nout), range(self.Nout)) if i[0]!=i[1]]
        for i, (r, s) in enumerate(receiver_sender_list):
            self.Rr_dec[r, i] = 1
            self.Rs_dec[s, i] = 1
        self.Rr_dec = Variable(self.Rr_dec).cuda()
        self.Rs_dec = Variable(self.Rs_dec).cuda()

    def INlayer(x, Rr, Rs, fr, fo, N, P, Nr, De, Do):
        Orr = self.tmul(x, Rr)
        Ors = self.tmul(x, Rs)
        B = torch.cat([Orr, Ors], 1)
        ### First MLP ###
        B = torch.transpose(B, 1, 2).contiguous()
        B = nn.functional.relu(fr[0](B.view(-1, 2 * P)))
        B = nn.functional.relu(fr[1](B))
        E = nn.functional.relu(fr[2](B).view(-1, Nr, De))
        del B
        E = torch.transpose(E, 1, 2).contiguous()
        Ebar = self.tmul(E, torch.transpose(Rr, 0, 1).contiguous())
        del E
        C = torch.cat([x, Ebar], 1)
        del Ebar
        C = torch.transpose(C, 1, 2).contiguous()
        C = nn.functional.relu(fo[0](C.view(-1, P + De)))
        C = nn.functional.relu(fo[1](C))
        # force outputs in [0,1]
        O = nn.functional.sigmoid(fo[2](C).view(-1, N, Do))
        del C
        return O
        
    def forward(self, x):
        # encoder 
        O_Encoder = INlayer(x, self.Rr_enc, self.Rs_enc,
                             [self.fr1_enc, self.fr2_enc, self.fr3_enc],
                             [self.fo1_enc, self.fo2_enc, self.fo3_enc],
                             self.N_in, self.P, self.Nr_in, self.De, self.Do)
        # now sum over the N_in vertices to obtain the Do latent quantities
        O_Encoder = torch.sum(O_Encoder, 1)
        # add variational here?
        # NOT FOR NOW

        # create a N_out x Do tensor, to be given as input to the decoder
        In_decoder = nn.functional.relu(self.ConvTranspose(O_Encoder))
        #In_decoder = In_decoder.view(-1, self.N_out, self.Do)

        # decoder
        # we use the same De as the encoder (not needed but ...)
        O_Decoder = INlayer(In_decoder, self.Rr_dec, self.Rs_dec,
                             [self.fr1_dec, self.fr2_dec, self.fr3_dec],
                             [self.fo1_dec, self.fo2_dec, self.fo3_dec],
                             self.N_out, self.Do, self.Nr_out, self.De, self.P)

        # the oputput needs some manipulation
        # 1) replace the next-to-last feature (PFClass) with floor(PFClass*5)
        # 2) replace the last feature (PFClassCharge) with floor(Charge*3)

        O_Decoder[:,:6] = torch.floor(O_Decoder[:,:6]*5.)
        O_Decoder[:,:7] = torch.floor(O_Decoder[:,:7]*3.)

        # TODO: 3) if PFClass==1 or ==2 -> Charge (last feature = 0, sign(Charge-0.5) otherwise

        return 
        
    def tmul(self, x, y):  #Takes (I * J * K)(K * L) -> I * J * L 
        x_shape = x.size()
        y_shape = y.size()
        return torch.mm(x.view(-1, x_shape[2]), y).view(-1, x_shape[1], y_shape[1])

####################
    
def get_sample(training, target, choice):
    target_vals = np.argmax(target, axis = 1)
    ind, = np.where(target_vals == choice)
    chosen_ind = np.random.choice(ind, 50000)
    return training[chosen_ind], target[chosen_ind]

def accuracy(predict, target):
    _, p_vals = torch.max(predict, 1)
    r = torch.sum(target == p_vals.squeeze(1)).data.numpy()[0]
    t = target.size()[0]
    return r * 1.0 / t

def stats(predict, target):
    print(predict)
    _, p_vals = torch.max(predict, 1)
    t = target.cpu().data.numpy()
    p_vals = p_vals.squeeze(1).data.numpy()
    vals = np.unique(t)
    for i in vals:
        ind = np.where(t == i)
        pv = p_vals[ind]
        correct = sum(pv == t[ind])
        print("  Target %s: %s/%s = %s%%" % (i, correct, len(pv), correct * 100.0/len(pv)))
    print("Overall: %s/%s = %s%%" % (sum(p_vals == t), len(t), sum(p_vals == t) * 100.0/len(t)))
    return sum(p_vals == t) * 100.0/len(t)


def my_loss(output, target):
    # standard MAE 

# ### Prepare Dataset
nVertices = 100
x = []
x.append(50) # hinned nodes
x.append(12) # De
x.append(4) # Do

#####
val_split = 0.3
batch_size = 100
n_epochs = 100
patience = 10

######
labels = ['Pt', 'Eta', 'Phi', 'ChPFIso', 'GammaPFIso', 'NeuPFIso', 'PfClass', 'Charge']

import glob
inputTrainFiles = glob.glob("../data/*.h5" %nParticles)
inputValFiles = inputTrainFiles[:, int(len(inputTrainFiles)*val_split)]
inputTrainFiles = inputTrainFiles[int(len(inputTrainFiles)*val_split),:]
#inputTrainFiles = glob.glob("/data/ml/mpierini/hls-fml/jetImage*_%sp*.h5" %nParticles)
#inputValFiles = glob.glob("/data/ml/mpierini/hls-fml/VALIDATION/jetImage*_%sp*.h5" %nParticles)

mymodel = GraphNet(nParticles, len(labels), params, int(x[0]), int(x[1]), int(x[2]), 0)
optimizer = optim.Adam(mymodel.parameters(), lr = 0.0001)

loss_train = np.zeros(n_epochs)
loss_val = np.zeros(n_epochs)
nBatches_per_training_epoch = len(inputTrainFiles)*10000/batch_size
nBatches_per_validation_epoch = len(inputValFiles)*10000/batch_size
print("nBatches_per_training_epoch: %i" %nBatches_per_training_epoch)
print("nBatches_per_validation_epoch: %i" %nBatches_per_validation_epoch)
for i in range(n_epochs):
    if mymodel.verbose: print("Epoch %s" % i)
    # Define the data generators from the training set and validation set.
    random.shuffle(inputTrainFiles)
    random.shuffle(inputValFiles)
    train_set = InEventLoader(file_names=inputTrainFiles, nP=nParticles,
                              feature_name ='jetConstituentList',label_name = 'jets', verbose=False)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=False)
    val_set = InEventLoader(file_names=inputValFiles, nP=nParticles,
                            feature_name ='jetConstituentList',label_name = 'jets', verbose=False)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False)
    ####
    # train
    for batch_idx, mydict in enumerate(train_loader):
        data = mydict['jetConstituentList']
        target = mydict['jets']
        if args_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data), Variable(target)
        optimizer.zero_grad()
        out = mymodel(data)
        l = my_loss(out, target)
        l.backward()
        optimizer.step()
        loss_train[i] += l.cpu().data.numpy()/nBatches_per_training_epoch
    # validation
    for batch_idx, mydict in enumerate(val_loader):
        data = mydict['jetConstituentList']
        target = mydict['jets']
        if args_cuda:
            data, target = data.cuda(), target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        out_val = mymodel(data)
        l_val = my_loss(out_val, target)
        loss_val[i] += l_val.cpu().data.numpy()/nBatches_per_validation_epoch
    if mymodel.verbose: print("Training   Loss: %f" %loss_train[i])
    if mymodel.verbose: print("Validation Loss: %f" %loss_val[i])
    if all(loss_val[max(0, i - patience):i] > min(np.append(loss_val[0:max(0, i - patience)], 200))) and i > patience:
        print("Early Stopping")
        break

# save training history
import h5py
f = h5py.File("%s/history.h5" %sys.argv[1], "w")
f.create_dataset('train_loss', data= np.asarray(loss_train), compression='gzip')
f.create_dataset('val_loss', data= np.asarray(loss_val), compression='gzip')

# the best model
torch.save(mymodel.state_dict(), "%s/IN_bestmodel.params" %(sys.argv[1]))
