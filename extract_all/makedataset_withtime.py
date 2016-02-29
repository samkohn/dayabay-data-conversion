###############################3######
# Peter Sadowski 2015
# Make supervised learning dataset
####################################333

import os
import sys
#import h5py
from collections import defaultdict
import roottools
from roottools import ismuon, isflasher
import array
import numpy as np
import itertools
import pickle as pkl
#import pandas
import h5py
import time
import argparse

NFEATURES = 192 * 2

def main():
    #print "usage: python large_make-dif.py num localFid globalFid"
    #num is the total number of events that tobe extracted
    #localfid is within a taskfarmer run, e.g., if we run 20 tasks in a single taskfarmer master run, then localid is between 0-19
    #globalfid is id between different taskfarmer run, e.g.,if we have 2 taskfarmer master run, then the first run has globalfid as 0, and the second one starts after the total number of first tasks, which is 20
    if len(sys.argv)<3:
        print "follow usage.."
        sys.exit(1)
    #open Filelist and pick the the file
    with open("./FileList-6Oct-Official-1") as f:
        content = [x.strip('\n') for x in f.readlines()]
    N = int(sys.argv[1]) # Number of examples from each class.20k
    fileId=int(sys.argv[2])
#    N = 8* 10**5 # Number of examples from each class.
    nclasses = 5
    classnames = ['ad_init', 'ad_delay', 'muon', 'flasher', 'other']
    data = {}
    for name in classnames:
        data[name] = np.zeros((N, NFEATURES), dtype='float32')

    # AD data
#    fn = '/global/homes/p/pjsadows/ad/ibd_%d.h5' % N
    Nad = 16 + fileId
    fn = '/project/projectdirs/das/wbhimji/mantissa-hep/dayabay/data/ad/ibd_v3_time/%d.h5' % Nad
    f = h5py.File(fn, 'r')
    charges = f['charges']
    data['ad_init'] = charges[:N, 0:NFEATURES] # change this for new files with 2* nfeatures
    data['ad_delay'] = charges[:N, NFEATURES:2*NFEATURES]
    f.close()   
 
    # Muon, flasher, other.
    counts = {'muon':0, 'flasher':0, 'other':0}
#    rootfile = 'a/project/projectdirs/dayabay/scratch/ynakajim/mywork/data_samples/recon.Neutrino.0021221.Physics.EH1-Merged.P14A-P._0001.root'
    rootfile = content[fileId]
    for entry in roottools.rootfileiter(rootfile):
        if entry['detector'] not in [0,1,2,3]:
            continue # Not AD detector
        charge,time = roottools.getChargesTime(entry, preprocess_flag=True)
        if ismuon(entry):
            counts['muon'] += 1
            if counts['muon'] <= N:
                data['muon'][counts['muon']-1,:] = np.hstack((charge.flatten(),time.flatten()))
        elif isflasher(entry):
            counts['flasher'] += 1
            if counts['flasher'] <= N:
                data['flasher'][counts['flasher']-1,:] =  np.hstack((charge.flatten(),time.flatten()))
        else:
            counts['other'] += 1 # TODO: should we ignore post-muon noise?
            if counts['other'] <= N:
                data['other'][counts['other']-1,:] =  np.hstack((charge.flatten(),time.flatten()))

        if np.all(np.array(counts.values()) > N):
            print np.array(counts.values())
            print np.array(counts.values()) > N
            break

    print counts

    X = np.vstack([data[name] for name in classnames])
    Y = np.zeros((nclasses*N, nclasses), dtype='float32')
    #Y = np.arange(nclasses*N, dtype='int32') / N #
    for i in range(nclasses*N):
        label = i / N
        Y[i,label] = 1.0        
 
    if False:
        # Randomize examples.
        perm = np.random.permutation(N*nclasses)
        X = X[perm, :]
        Y = Y[perm]
        
    # Write
#    outfile = 'single_withtime_%d.h5' % N
    outfile = "./single_withtime_v2_"+str(fileId)+".h5"
    f = h5py.File(outfile, "w")
    dset = f.create_dataset("inputs", X.shape, dtype='float32')
    dset2 = f.create_dataset("targets", Y.shape, dtype='float32')
    dset[0:N*nclasses,...] = X
    dset2[0:N*nclasses] = Y
    f.close()
    #np.savetxt('single_%d.csv' % N, np.hstack([Y.reshape((-1,1)), X]))
    #np.savetxt('single_%d.csv' % N, np.hstack([Y, X]))


if __name__=='__main__':
    main()
