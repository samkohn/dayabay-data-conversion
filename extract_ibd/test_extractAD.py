###############################3######
# Peter Sadowski 2015
# Extract AD candidates
####################################333

import os
import sys
#import h5py
from collections import defaultdict
import roottools
import array
import numpy as np
import itertools
import pickle as pkl
import pandas
import h5py
import thread

NFEATURES = 11 + 4*192

def main():
    
    filename = '/global/homes/p/pjsadows/data/dayabay/ibd_candidates_eh1.txt' # Files containing list of AD candidates.
    X = pandas.read_csv(filename, delimiter='\t')
    fdict = pkl.load(open('fdict_wahid_v2.pkl', 'r')) # Contains map from (run, filenum) to full filepath.
    Nstart = int( sys.argv[1]) 
    start = Nstart * 1  * 10**4 # 10k events per file 
#    stop =  (Nstart + 1) * 1 * 10**4 - 1  #10
#    start = 1
#    stop =  2
    N = 1 * 10**4
#    stop = start + N 
    outfile = 'ibd_v4_time_%d.h5' % start
    #outfile = 'trial_ibd_%d.h5' % N
    data = np.zeros((N, NFEATURES), dtype='f4')
    i = -1
    for j,trigger in X.iterrows():
        if (trigger['RunNo'], trigger['FileNo']) not in fdict:
            print 'Error: Could not find run %d file %d' % (trigger['RunNo'], trigger['FileNo'])
            continue
        fn = fdict[(trigger['RunNo'], trigger['FileNo'])]
        if not os.access(fn, os.R_OK):
            print 'Error: Could not read file %s' % fn
            continue
        i += 1
        if i < start:
            continue
        if i >= stop:
            break # Early
        try:
            print fn
            rval = extract_candidate(fn, [trigger['Detector']+1, trigger['Detector']+1], [trigger['trigno_prompt'], trigger['trigno_delayed']])
        except:
            print 'Error: Could not load trees from %s' % fn
            i -= 1
            continue
        data1, data2 = rval[0], rval[1]
        data[i - start, :] = np.hstack((trigger.values.reshape((1,-1)), data1, data2))
    if False:
        pkl.dump(data, outfile)     
    else:
        f = h5py.File(outfile, "w")
        dset = f.create_dataset("charges", (N, 4*192), dtype='float32')
        dset2 = f.create_dataset("info", (N, 11), dtype='float32')
        dset[0:N, ...] = data[:, 11:]
        dset2[0:N, ...] = data[:, :11]
        f.close()
#        #h5file, node = init_hdf5(filename, ([nexamples, ninputs], [nexamples, noutputs]))
        #fill_hdf5(h5file, data_x=X, data_y=Y, node=node, start=0)

def extract_candidate(filename, detectors, triggerNumbers):   
 
    treename = '/Event/CalibReadout/CalibReadoutHeader'
    intbranches = ['nHitsAD','triggerNumber', 'detector']
    floatbranches = []
    ivectorbranches = ["ring","column","wallNumber"] #,"wallspot"]
    fvectorbranches = ["timeAD","chargeAD", "timePool", "chargePool", "wallSpot"]
    t1 = roottools.RootTree(filename, treename, intbranches=intbranches, floatbranches=floatbranches, ivectorbranches=ivectorbranches, fvectorbranches=fvectorbranches)
 
    #treename = '/Event/Data/CalibStats'
    #floatbranches = ['flasher_flag', 'MaxQ', 'Quadrant', 'time_PSD', 'time_PSD1', 'MaxQ_2inchPMT'] #, 'dtLast_AD1_ms', 'dtLast_AD2_ms', 'dtLast_AD3_ms', 'dtLast_AD4_ms'] 
    #intbranches = ['triggerNumber']#["detector","triggerNumber"] #,"triggerType","triggerTimeSec","triggerTimeNanoSec","nHitsAD","nHitsPool"]    
    #ivectorbranches = []
    #fvectorbranches = []
    #t2 = roottools.RootTree(filename, treename, intbranches=intbranches, floatbranches=floatbranches, ivectorbranches=ivectorbranches, fvectorbranches=fvectorbranches)
    #requirements = [('detector', detector), ('triggerNumber', triggernum)]

    startidx = 0 #triggerNumbers[0]
    rval = []
    for (detector, triggerNumber) in zip(detectors, triggerNumbers): 
        idx = t1.find_trigger(detector, triggerNumber, startidx)
        assert idx is not None, filename
        e1 = t1.loadentry(idx)
        charge, time = roottools.getChargesTime(e1, preprocess_flag=True)
        #isflasher = roottools.isflasher(e2)       
        data = np.zeros((1,2*192), dtype='float32')
        data[0, :] =  np.hstack((charge.flatten(),time.flatten()))
        rval.append(data)
        startidx = idx
    return rval


if __name__=='__main__':
    main()
