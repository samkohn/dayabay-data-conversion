"""
This script will create a sample of accidental IBD pairs (uncorrelated
background) that is statistically similar to the true accidental background
that contaminates the IBD sample.

Info on software requirements. This software requires ROOT 6, specifically
version 6.06.04. It currently only works on Edison (and not on Cori).

"""
import roottools as rt
from hitmapformat import METADATA_NAMES, NPIXELS, NMETADATA, ENTRYSIZE
import hitmapformat as formatter
import matplotlib.pyplot as plt
import h5py
import sys
import os
import re
import numpy as np
import itertools
import argparse
import logging
from itertools import chain

def prepareEventDataForH5(prompt, delayed, dt, label):
    """
        Return a numpy array with the relevant data for saving to HDF5.

        This method makes use of the hitmapformat.py module.
    """
    event = {}
    event['runno'] = prompt['runno']  # from original file IO
    event['fileno_prompt'] = prompt['fileno']  # from original file IO
    event['site'] = prompt['site']
    event['det'] = prompt['detector']
    event['time_sec'] = prompt['triggerTimeSec']
    event['time_nanosec'] = prompt['triggerTimeNanoSec']
    event['trigno_prompt'] = prompt['triggerNumber']
    event['energy_prompt'] = prompt['energy']
    event['dt_last_ad_muon'] = prompt['dtLast_ADMuon_ms']
    event['dt_last_ad_shower_muon'] = prompt['dtLast_ADShower_ms']
    event['dt_last_wp_muon'] = min(prompt['dtLastIWS_ms'],
            prompt['dtLastOWS_ms'])
    event['dt_IBD'] = dt
    event['label_id'] = label
    charge_prompt, time_prompt = rt.getChargesTime(prompt)
    if delayed is not None:
        event['fileno_delayed'] = delayed['fileno']
        event['energy_delayed'] = delayed['energy']
        event['trigno_delayed'] = delayed['triggerNumber']
        charge_delayed, time_delayed = rt.getChargesTime(delayed)
    else:
        event['fileno_delayed'] = 0
        event['energy_delayed'] = 0
        event['trigno_delayed'] = 0
        charge_delayed, time_delayed = None, None
    flattened = formatter.getFlattenedData(event,
            charge_prompt, time_prompt, charge_delayed, time_delayed)
    return flattened

def bulk_update(first, *args):
    """
        An extension of the dict.update which updates one dict with many more
        dicts.

        Like the normal dict.update, returns None.

    """
    for other in args:
        first.update(other)

    return None

def get_root_file_info(name):
    """Extract the run number and file number from the ROOT file name.

    The file name must match the following regular expression, otherwise
    a ValueError is thrown:

    /[A-Za-z.]+(\d+)[A-Za-z0-9.-]+_(\d+)\.root\.(prompt|delayed|flasher)\.root/
        ^        ^         ^         ^       ^
    prefix    run number  other  file number  suffix

    """
    expression = (r"[A-Za-z.]+(\d+)[A-Za-z0-9.-]+_(\d+)\.root" +
        r"\.(prompt|delayed|flasher)\.root")
    basename = os.path.basename(name)
    match = re.match(expression, basename)
    logging.debug("basename = %s", basename)
    if match:
        return map(int, match.group(1, 2))
    else:
        raise ValueError("Could not parse file name %s" % name)

def get_EHxADy_string(data):
    """Construct an ID string of the form "EH1AD2" or similar based on the
    given data from a TTree."""
    detector = data['detector']
    site = data['site']
    detectors = {n: 'AD%s' % n for n in range(1, 5)}
    sites = {1: 'EH1', 2: 'EH2', 4: 'EH3'}
    return '%s%s' % (sites[site], detectors[detector])

def get_data_for_file(rootfilename, max_events):
    """Returns a dict of lists for the specified file. The keys of the
    dict are the EHxADy strings labeling where the events in that list
    happened."""
    runno, fileno = get_root_file_info(rootfilename)
    fileinfodict = {'runno': runno, 'fileno': fileno}
    try:
        readout = rt.makeCalibReadoutTree(rootfilename)
        stats = rt.makeCalibStatsTree(rootfilename)
        rec = rt.makeRecTree(rootfilename)
    except:
        logging.info("error in file %s, skipping...", rootfilename)
        return {'EH1AD1':[], 'EH1AD2':[]}

    num_triggers = readout.numEntries()
    assert num_triggers == stats.numEntries(), "len(readout) != len(stats)"
    assert num_triggers == rec.numEntries(), "len(readout) != len(rec)"
    logging.info("num triggers = %d", num_triggers)
    if max_events < 0:
        max_events = num_triggers
    events = {'EH1AD1':[], 'EH1AD2':[]}
    for i, (readout_data, stats_data, rec_data) in enumerate(itertools.izip(readout.getentries(),
            stats.getentries(), rec.getentries())):
        if len(events['EH1AD1']) + len(events['EH1AD2']) >= max_events:
            break
        all_data = {}
        readout_data.unlazyconstruct()
        stats_data.unlazyconstruct()
        rec_data.unlazyconstruct()
        bulk_update(all_data, readout_data, stats_data, rec_data,
                fileinfodict)
        try:
            label = get_EHxADy_string(all_data)
        except KeyError:
            continue
        if label in events.keys():
            events[label].append((i, all_data))
        else:
            events[label] = [(i, all_data)]
    return events

def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', help='name of output h5 file')
    parser.add_argument('--infiles',
            help='a text file listing all of the files to draw events from')
    parser.add_argument('--max-prompt', type=int, default=-1,
            help='max number of prompt events to read (-1 = all)')
    parser.add_argument('--max-delayed', type=int, default=-1,
            help='max number of delayed events to read (-1 = all)')
    parser.add_argument('-l', '--label', type=str, required=True,
            choices=['flasher', 'accidental'],
            help='label for all events processed')
    parser.add_argument('-d', '--debug', action='store_true')
    return parser

def fetch_accidentals(args):
    prompts_to_fetch = args.max_prompt
    delayeds_to_fetch = args.max_delayed
    all_prompts = {}
    all_delayeds = {}
    for rootfile in rootfilenames:
        filename = rootfile + '.prompt.root'
        file_prompts = get_data_for_file(filename, prompts_to_fetch)
        filename = rootfile + '.delayed.root'
        file_delayeds = get_data_for_file(filename, delayeds_to_fetch)
        # Update/extend the all-prompts and all-delayeds dicts/lists
        # to include the newly retrieved event data. Update key-by-key to keep
        # different ADs separate.
        new_prompts = 0
        for key, value in file_prompts.iteritems():
            new_prompts += len(file_prompts[key])
            if key in all_prompts.keys():
                all_prompts[key].extend(file_prompts[key])
            else:
                all_prompts[key] = file_prompts[key]
        new_delayeds = 0
        for key, value in file_delayeds.iteritems():
            new_delayeds += len(file_delayeds[key])
            if key in all_delayeds.keys():
                all_delayeds[key].extend(file_delayeds[key])
            else:
                all_delayeds[key] = file_delayeds[key]
        # Update the number of events left to fetch
        if prompts_to_fetch >= 0:
            continue_prompt = prompts_to_fetch > new_prompts
            prompts_to_fetch -= new_prompts
        else:
            continue_prompt = True
        if delayeds_to_fetch >= 0:
            continue_delayed = delayeds_to_fetch > new_delayeds
            delayeds_to_fetch -= new_delayeds
        else:
            continue_delayed = True
        if (not continue_prompt) and (not continue_delayed):
            break

    # Algorithm: Shuffle the two lists' orders. Then pair up events. TODO: As a
    # trivial safety measure, ensure that no event is paired up with itself.
    all_prompts = {key: np.array(val) for key, val in all_prompts.iteritems()}
    all_delayeds = {key: np.array(val) for key, val in
            all_delayeds.iteritems()}
    for eventlist in all_prompts.values():
        np.random.shuffle(eventlist)
    for eventlist in all_delayeds.values():
        np.random.shuffle(eventlist)
    # Note: python's zip method discards any unpaired events. This is fine in
    # our case since there's no default and we don't want repeats.
    pairs = []
    for key in all_delayeds.keys():
        delayeds = all_delayeds[key]
        if key in all_prompts.keys():
            prompts = all_prompts[key]
            pairs.extend(zip(prompts, delayeds))
        else:
            pass
    num_pairs = len(pairs)
    dset_to_save = np.empty((num_pairs, ENTRYSIZE), dtype=float)

    # The expected distribution of dts between prompt and delayed for
    # uncorrelated signals is an exponential with a time constant of 1/rate.
    # Since this time (20ms) is much greater than our time cut (200us) by a
    # factor of 100, the exponential distribution is closely approximated by a
    # uniform distribution. (The PDFs differ by one part in one hundred near the
    # time cut region.) Using a uniform distribution is faster since I can
    # control the domain of the PDF without manually truncating the
    # distribution.
    DT_THRESHOLD = 0.2  # ms
    dts = np.random.uniform(0, DT_THRESHOLD, (len(pairs),))

    for i, (dt, ((j, prompt), (k, delayed))) in enumerate(zip(dts, pairs)):
        dset_to_save[i, :] = prepareEventDataForH5(prompt, delayed, dt,
                args.label)
    return dset_to_save

def fetch_flashers(args):
    '''fetch flashers from the specified files and save them as 'singles' (no
    delayed trigger).'''
    flashers_to_fetch = args.max_prompt
    all_flashers = {}
    for rootfile in rootfilenames:
        filename = rootfile + '.flasher.root'
        file_flashers = get_data_for_file(filename, flashers_to_fetch)

        new_flashers = 0
        for key, value in file_flashers.iteritems():
            new_flashers += len(file_flashers[key])
            if key in all_flashers.keys():
                all_flashers[key].extend(file_flashers[key])
            else:
                all_flashers[key] = file_flashers[key]

        if flashers_to_fetch >= 0:
            continue_flasher = flashers_to_fetch > new_flashers
            flashers_to_fetch -= new_flashers
        else:
            continue_flasher = True
        if not continue_flasher:
            break
    flashers_list = list(chain(*all_flashers.values()))
    num_pairs = len(flashers_list)
    dset_to_save = np.empty((num_pairs, ENTRYSIZE), dtype=float)
    for i, (j, flasher) in enumerate(flashers_list):
        dset_to_save[i, :] = prepareEventDataForH5(flasher, None, 0, args.label)
    return dset_to_save

if __name__ == "__main__":
    parser = setup_parser()
    args = parser.parse_args()
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)
    outfilename = 'accidentals.h5' if args.output is None else args.output
    h5file = h5py.File(outfilename, "w")

    if args.infiles is None:
        runno = '0021221'
        fileno = '0016'
        rootfilenames = [("/global/project/projectdirs/dayabay/data/exp/" +
            "dayabay/2011/p14a/Neutrino/1224/recon.Neutrino.%s." +
            "Physics.EH1-Merged.P14A-P._%s.root") % (runno, fileno)]
    else:
        with open(args.infiles, 'r') as f:
            rootfilenames = map(str.strip, f.readlines())

    if args.label == 'flasher':
        dset_to_save = fetch_flashers(args)
    elif args.label == 'accidental':
        dset_to_save = fetch_accidentals(args)

    outdset = h5file.create_dataset("accidentals_bg_data",
            data=dset_to_save, compression="gzip", chunks=True)

    outdset.attrs['description'] = \
"""This dataset contains accidental background pairs that have been created
artificially by pairing up singles triggers.

The selection criteria are the same as the IBD selection criteria for
prompt-like singles and delayed-like singles, with the only difference being
the multiplicity cut, which rejects candidates that are within 400us before or
after any other event with >0.7 MeV.

See the 'structure' attribute for a detailed description of the dset layout.

Each row in the dataset contains flattened versions of the charge and time on
each PMT for both the prompt and delayed events, followed by some metadata.
There are currently %d metadata items, which can be accessed as attributes from
0 to %d. Reverse lookup is also implemented to get the index of a given piece
of metadata.""" % (NMETADATA, NMETADATA)
    outdset.attrs['structure'] = \
"""Data set structure: N rows by %d columns. Each column has the following:

0-%d: flattened 8x24 of prompt charge
%d-%d: flattened 8x24 of prompt time
%d-%d: flattened 8x24 of delayed charge
%d-%d: flattened 8x24 of delayed time
%d-%d: metadata such as trigger numbers, site and AD info, and dt between
triggers and to the last muon.""" % (ENTRYSIZE, NPIXELS-1, NPIXELS,
    2*NPIXELS-1, 2*NPIXELS, 3*NPIXELS-1, 3*NPIXELS, 4*NPIXELS-1, 4*NPIXELS,
    ENTRYSIZE-1)
    # Set metadata attributes so people know what's in the last bit of each row
    for i, name in enumerate(METADATA_NAMES[formatter.VERSION]):
        outdset.attrs[str(i)] = name
        outdset.attrs[name] = i

    h5file.close()
    print "Exiting..."
