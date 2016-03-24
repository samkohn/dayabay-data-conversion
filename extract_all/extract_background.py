
# coding: utf-8

# In[1]:

###############################3######
# Peter Sadowski 2015
# Make supervised learning dataset
####################################333

import os
import sys
#import h5py
from collections import defaultdict
import roottools #move to /global/common
from roottools import ismuon, isflasher
import array
import numpy as np
import itertools
import pickle as pkl
#import pandas
import h5py
import time
import argparse
import itertools
import pandas


# In[ ]:

taskfarmer_id = int(sys.argv[1])


# In[2]:

NFEATURES = 192


# In[3]:

filename = '/global/homes/p/pjsadows/data/dayabay/ibd_candidates_eh1.txt' # Files containing list of AD candidates.
X = pandas.read_csv(filename, delimiter='\t')


# In[4]:

# 1 
event_dict = {'ibd_prompt':1,
'ibd_delay':2, 
'muon':3,
'flasher':4, 
'other':5}


# In[84]:

def is_ibd(entry, file_no, run_no):
        ibds = X[(X['RunNo']==run_no) & (X['FileNo']==file_no) ]
        ibd_prompt_cand = any(ibds[ibds['trigno_prompt'] == entry['triggerNumber']].any())
        ibd_delay_cand = any(ibds[ibds['trigno_delayed'] == entry['triggerNumber']].any())
        if ibd_prompt_cand and ibd_delay_cand:
            assert False, "Labelled as both prompt and delay. Huh?"
        elif ibd_prompt_cand:
            ret = 'ibd_prompt'
        elif ibd_delay_cand:
            ret = 'ibd_delay'
        else:
            ret = False
        return ret
        


# In[6]:

def get_background_type(entry):
    if ismuon(entry):
        return event_dict['muon']
    elif isflasher(entry):
        return event_dict['flasher']
    else:
        return event_dict['other']


# In[7]:

def get_class(entry, file_no, run_no):
    ibd_name = is_ibd(entry, file_no, run_no)
    if ibd_name:
        return event_dict[ibd_name]
    else:
        return get_background_type(entry)


# In[8]:

def get_eh(rootfile):
    fs_split = rootfile.split('.')
    return fs_split[4].split('-')[0]


# In[9]:

def get_run_no(file_string):
    fs_split = file_string.split('.')
    return int(fs_split[2])
def get_file_no(file_string):
    fs_split = file_string.split('.')
    return int(fs_split[6].split('_')[1])


# In[10]:


#with open("./FileList-6Oct-Official-1") as f:
#with open("/global/homes/r/racah/projects/dayabay-data-conversion/extract_all/FileList-14Mar-Recovered-1-2", "r") as f:
with open("/global/homes/r/racah/projects/dayabay-data-conversion/extract_all/Unprocessed_FileList-22Mar-1-2", "r") as f:
   content = [x.strip('\n') for x in f.readlines()]


# In[67]:

rootfile = content[taskfarmer_id]
run_no = get_run_no(rootfile)
file_no = get_file_no(rootfile)
eh = int(get_eh(rootfile)[2:])


# In[112]:

index=0
stat_entries = roottools.get_num_stat_entries(rootfile)
calib_entries =roottools.get_num_readout_entries(rootfile)
entries = {}
t1 = time.time()

#these two for loops will be slow
# make a hash table mapping triggerNumber to charge, time info entry
for entry1 in roottools.calibReadoutIter(rootfile):
    if entry1['detector']  in [0,1,2,3,4]:
        entries[entry1['triggerNumber']] = entry1
t2 = time.time()
print "it took %d seconds for %i events. Thats %i events per second" % (t2-t1, calib_entries, calib_entries / (t2-t1))

t1 = time.time()
#if a flasher stat entry has the same triggerNumber as a readout entry, merge them
for entry2 in roottools.calibStatsIter(rootfile):
    if entry2['triggerNumber'] in entries:
        entries[entry2['triggerNumber']].update(entry2)
t2 = time.time()
print "it took %d seconds for %i events. Thats %i events per second" % (t2-t1, stat_entries, stat_entries / (t2-t1))
#now go thru and so normal parsing (this should be quick)


# In[109]:

num_entries = len(entries)
dataset_keys = ['class', 'charge', 'time', 'trig_no', 'detector_no']
data = {}
for k in dataset_keys:
    if k == 'charge' or k=='time':
        data[k] = np.zeros((num_entries,NFEATURES), dtype="float64") #h5py file here?
    else:
        data[k] = np.zeros((num_entries,1), dtype='int32')
data['run_no'] = run_no * np.ones((num_entries,1), dtype='int32')
data['file_no'] = file_no * np.ones((num_entries,1), dtype='int32')
data['eh'] = eh * np.ones((num_entries,1), dtype='int32')


# In[113]:

index=0
for entry in entries.values():
#     if entry['detector'] not in [0,1,2,3]:
#         continue # Not AD detector 
    d ={}
    d['charge'],d['time'] = roottools.getChargesTime(entry, preprocess_flag=False, dtype='float64')
    if index % 1000 == 0:
        print "%i of %i entries" % (index, len(entries))
    #flatten the 8,24 arrays
    for k in ['charge', 'time']:
        d[k] = d[k].flatten()
    d['trig_no'] = entry['triggerNumber']
    d['detector_no'] = entry['detector']
    d['class'] = get_class(entry,file_no, run_no)
    for k,v in d.iteritems():
        data[k][index] = v
    index += 1
 


# In[115]:

h5_filename = 'recon.' + rootfile.split('.root')[0].split('recon.')[1] + '.h5'
print h5_filename
path = '/project/projectdirs/paralleldb/spark/benchmarks/nmf/daya-data'#  if len(sys.argv) < 3 else sys.argv[2]
h5_path = os.path.join(path, h5_filename)
print h5_path


# In[116]:

h5f = h5py.File(h5_path, 'w')
for k,v in data.iteritems():
     h5f.create_dataset(k,data=v[:index]) # this is importatn to get rid of zero rows
h5f.close()
os.chown(h5_path,61228,70018) #changes file to be owned by racah and in group dasrepo

