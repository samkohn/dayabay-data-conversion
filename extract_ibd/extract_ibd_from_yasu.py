###############################3######
# Sam Kohn 2016
# Extract IBD candidates
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
import logging
logging.basicConfig(level=logging.DEBUG)

NFEATURES = 11 + 4*192

def main():
    
    filelistname = 'yasufiles.txt' # ROOT files containing IBD candidates
    treename = 'tr_ibd'
    filelist = pandas.read_csv(filelistname, squeeze=True, header=None)
    Nstart = int( sys.argv[1]) 
    N = 10000 if len(sys.argv)== 2 else int(sys.argv[2])  # 10k events per file 
    start = Nstart * N
    stop = start + N 
    outfile = 'ibd_v4_time_%d.h5' % start
    data = np.zeros((N, NFEATURES), dtype='f4')
    # find starting value
    current_index = 0
    for i, filename in enumerate(filelist):
        roottree = roottools.RootTree(filename, treename)
        numentries = roottree.numEntries()
        if current_index + numentries >= start:
            startfile = filename
            startfileindex = i
            startentry = start - current_index
            break
        else:
            current_index += numentries
    # Go through the starting file
    intbranches = ['runno', 'fileno', 'site', 'det', 'time_sec',
        'time_nanosec', 'trigno_prompt', 'trigno_delayed',
        'nHitsAD_prompt', 'nHitsAD_delayed']
    floatbranches = ['dt_last_ad_muon', 'dt_last_ad_shower_muon',
        'dt_last_wp_muon']
    ivectorbranches = ['hitCountAD_prompt', 'ring_prompt', 'column_prompt',
        'hitCountAD_delayed', 'ring_delayed', 'column_delayed']
    fvectorbranches = ['timeAD_prompt', 'chargeAD_prompt', 'timeAD_delayed',
        'chargeAD_delayed']
    roottree = roottools.RootTree(startfile, treename, intbranches,
        floatbranches, ivectorbranches, fvectorbranches)
    current_index = start  # corresponds to range(startentry, ...)
    for entrynum in range(startentry, roottree.numEntries()):
        current_index += 1





    i = -1
    for j,trigger in X.iterrows():
        logging.debug('Start of loop')
        if (trigger['RunNo'], trigger['FileNo']) not in fdict:
            logging.info('Could not find run %d file %d',trigger['RunNo'], trigger['FileNo'])
            continue
        fn = fdict[(trigger['RunNo'], trigger['FileNo'])]
        if not os.access(fn, os.R_OK):
            logging.info('Could not read file %s', fn)
            continue
        i += 1
        if i < start:
            continue
        if i >= stop:
            break # Early
        try:
            logging.debug(fn)
            rval = extract_candidate(fn, [trigger['Detector']+1, trigger['Detector']+1], [trigger['trigno_prompt'], trigger['trigno_delayed']])
        except:
            logging.error('Error: Could not load trees from %s', fn)
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

def extract_candidate(filename, detectors, triggerNumbers):   
 
    treename = '/Event/CalibReadout/CalibReadoutHeader'
    intbranches = ['nHitsAD','triggerNumber', 'detector']
    floatbranches = []
    ivectorbranches = ["ring","column","wallNumber"]
    fvectorbranches = ["timeAD","chargeAD", "timePool", "chargePool", "wallSpot"]
    t1 = roottools.RootTree(filename, treename, intbranches=intbranches, floatbranches=floatbranches, ivectorbranches=ivectorbranches, fvectorbranches=fvectorbranches)
 
    startidx = 0
    rval = []
    for (detector, triggerNumber) in zip(detectors, triggerNumbers): 
        logging.debug('desired trigger = %d', triggerNumber)
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
