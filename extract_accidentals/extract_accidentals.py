"""
This script will create a sample of accidental IBD pairs (uncorrelated
background) that is statistically similar to the true accidental background
that contaminates the IBD sample.

Info on software requirements. This software requires ROOT 6, specifically
version 6.06.04. It currently only works on Edison (and not on Cori).

"""
import roottools as rt
from hitmapformat import *
import matplotlib.pyplot as plt
import h5py
import sys
import os
import re
import numpy as np
import itertools
import argparse
import logging

def passes_WS_muon_veto(readout_data, stats_data):
    """
       Veto and return false if within (-2, 600) us surrounding NHIT > 12 in
       IWS or OWS.

    """
    us_to_ms = 1e-3
    NEXT_DT = 2 * us_to_ms
    LAST_DT = 600 * us_to_ms
    dt_last_IWS = stats_data['dtLastIWS_ms']
    dt_last_OWS = stats_data['dtLastOWS_ms']
    dt_next_IWS = stats_data['dtNextIWS_ms']
    dt_next_OWS = stats_data['dtNextOWS_ms']
    return (dt_next_IWS > NEXT_DT and
            dt_last_IWS > LAST_DT and
            dt_next_OWS > NEXT_DT and
            dt_last_OWS > LAST_DT)

def passes_AD_muon_veto(readout_data, stats_data):
    """
        Veto and return false if (0, 1.4) ms after >3000 pe signal.

    """
    DT_THRESHOLD = 1.4
    dt = stats_data['dtLast_ADMuon_ms']
    return dt > DT_THRESHOLD

def passes_AD_shower_muon_veto(readout_data, stats_data):
    """
        Veto and return false if (0, 0.4) s after >3e5 pe signal.

    """
    DT_THRESHOLD = 400
    dt = stats_data['dtLast_ADShower_ms']
    return dt > DT_THRESHOLD

def passes_flasher_veto(readout_data, stats_data):
    """
        Veto and return false if fails the flasher veto.

        Defined as a flasher if:

        (Q3/(Q2+Q4))**2 + (Qmax/(Qtot * 0.45))**2 > 1

        (In the paper the criterion is that log10 of that quantity is
        greater than 0, but this is the same and slightly cheaper.)

    """
    Q2 = stats_data['QuadrantQ2']
    Q3 = stats_data['QuadrantQ3']
    Q4 = stats_data['QuadrantQ4']
    Qmax = stats_data['MaxQ']
    Qtot = stats_data['NominalCharge']
    SCALE = 0.45

    return (Q3 / (Q2 + Q4))**2 + (Qmax / Qtot / SCALE)**2 < 1

def has_prompt_energy(rec_data):
    """
        Return True if this event has prompt-like energy (0.7, 12) MeV.

    """
    energy = rec_data['energy']
    MIN_E = 0.7
    MAX_E = 12
    return energy > MIN_E and energy < MAX_E

def has_delayed_energy(rec_data):
    """
        Return True if this event has delayed_like energy (6, 12) MeV.

    """
    energy = rec_data['energy']
    MIN_E = 6
    MAX_E = 12
    return energy > MIN_E and energy < MAX_E

def is_IBD_trigger(readout_data):
    """
        Return True if the event is a trigger in an AD with the ESUM and NHIT
        triggers activated.

    """
    AD_codes = (1, 2, 3, 4)
    ESUM_code = 0x10001000
    NHIT_code = 0x10000100
    trigger_code = ESUM_code | NHIT_code
    detector = readout_data['detector']
    trigger = readout_data['triggerType']
    return (detector in AD_codes and
            trigger == trigger_code)

def is_prompt_like(readout_data, stats_data, rec_data):
    """
        Return True if the event data suggests it is prompt-like.

        The criteria for prompt-like are given in the Daya Bay 2016 Long Paper
        as:

           - Passes water shield muon veto

           - Passes AD muon veto

           - Passes AD shower muon veto

           - Passes flasher veto

           - Erec is between 0.7 MeV and 12 MeV

    """
    return (passes_WS_muon_veto(readout_data, stats_data) and
        passes_AD_muon_veto(readout_data, stats_data) and
        passes_AD_shower_muon_veto(readout_data, stats_data) and
        passes_flasher_veto(readout_data, stats_data) and
        has_prompt_energy(rec_data))

def is_delayed_like(readout_data, stats_data, rec_data):
    """
        Return True if the event data suggests it is delayed-like.

        The criteria for delayed-like are given in the Daya Bay 2016 Long Paper
        as:

           - Passes water shield muon veto

           - Passes AD muon veto

           - Passes AD shower muon veto

           - Passes flasher veto

           - Erec is between 6 and 12 MeV

    """
    return (passes_WS_muon_veto(readout_data, stats_data) and
        passes_AD_muon_veto(readout_data, stats_data) and
        passes_AD_shower_muon_veto(readout_data, stats_data) and
        passes_flasher_veto(readout_data, stats_data) and
        has_delayed_energy(rec_data))

def is_singles_like(readout_data, stats_data):
    """
        Return True if the event data suggests it is singles-like based only on
        other nearby IBD-like triggers.

        This is in essence a multiplicity cut:

          - last AD trigger and next AD trigger are both more than 400 us away

    """
    thisDetector = readout_data['detector']
    thisSite = readout_data['site']
    EH1 = 1
    EH2 = 2
    EH3 = 3
    if thisSite == EH1 or thisSite == EH2:
        ADs = [1, 2]
    elif thisSite == EH3:
        ADs = [1, 2, 3, 4]
    else:
        return
    if thisDetector in ADs:
        last_name = 'dtLastAD%d_ms' % thisDetector
        next_name = 'dtNextAD%d_ms' % thisDetector
    else:
        return
    exclusion_time = 400e-3
    # Some of the ADs are always set to -1 because they don't exist
    # This may mistakenly not veto the 0th event in each file which also has
    # ADs set to -1 since there is no previous trigger. I don't think that
    # matters.
    dt_last = stats_data[last_name]
    dt_next = stats_data[next_name]

    # Ensure that the time is larger than the exclusion time
    return (dt_last > exclusion_time and
            dt_next > exclusion_time)

def prepareEventDataForH5(prompt, delayed, dt):
    """
        Return a numpy array with the relevant data for saving to HDF5.

        This method makes use of the hitmapformat.py module.
    """
    event = {}
    event['runno'] = prompt['runno']  # from original file IO
    event['fileno'] = prompt['fileno']  # from original file IO
    event['site'] = prompt['site']
    event['det'] = prompt['detector']
    event['time_sec'] = prompt['triggerTimeSec']
    event['time_nanosec'] = prompt['triggerTimeNanoSec']
    event['trigno_prompt'] = prompt['triggerNumber']
    event['trigno_delayed'] = delayed['triggerNumber']
    event['dt_last_ad_muon'] = prompt['dtLast_ADMuon_ms']
    event['dt_last_ad_shower_muon'] = prompt['dtLast_ADShower_ms']
    event['dt_last_wp_muon'] = min(prompt['dtLastIWS_ms'],
            prompt['dtLastOWS_ms'])
    event['dt_IBD'] = dt
    charge_prompt, time_prompt = rt.getChargesTime(prompt)
    charge_delayed, time_delayed = rt.getChargesTime(delayed)
    flattened = getFlattenedData(event,
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

    /[A-Za-z.]+(\d+)[A-Za-z0-9.-]+_(\d+)\.root/
        ^        ^         ^         ^       ^
    prefix    run number  other  file number  suffix

    """
    expression = r"[A-Za-z.]+(\d+)[A-Za-z0-9.-]+_(\d+)\.root"
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

def get_prompt_delayed_like_data_for_file(rootfilename, max_prompts_desired,
        max_delayeds_desired):
    """Returns two dicts of lists, one of prompt-like and one of delayed-like singles
    for the specified file. The keys of the dict are the EHxADy strings
    labeling where the events in that list happened."""
    runno, fileno = get_root_file_info(rootfilename)
    fileinfodict = {'runno': runno, 'fileno': fileno}
    readout = rt.makeCalibReadoutTree(rootfilename)
    stats = rt.makeCalibStatsTree(rootfilename)
    rec = rt.makeRecTree(rootfilename)

    num_triggers = readout.numEntries()
    assert num_triggers == stats.numEntries(), "len(readout) != len(stats)"
    assert num_triggers == rec.numEntries(), "len(readout) != len(rec)"
    logging.info("num triggers = %d", num_triggers)
    if max_prompts_desired < 0:
        max_prompts_desired = num_triggers
    if max_delayeds_desired < 0:
        max_delayeds_desired = num_triggers
    # Determine which events are prompt-like and which ones are delayed-like
    # (allow for overlap: in particular, all delayed-like events are also
    # prompt-like).
    prompt_like_events = {}
    delayed_like_events = {}
    for i, (readout_data, stats_data, rec_data) in enumerate(itertools.izip(readout.getentries(),
            stats.getentries(), rec.getentries())):
        if (len(prompt_like_events) >= max_prompts_desired and
                len(delayed_like_events) >= max_delayeds_desired):
            break
        if is_IBD_trigger(readout_data) and is_singles_like(readout_data, stats_data):
            if (len(prompt_like_events) < max_prompts_desired and
                    is_prompt_like(readout_data, stats_data, rec_data)):
                all_data = {}
                readout_data.unlazyconstruct()
                stats_data.unlazyconstruct()
                rec_data.unlazyconstruct()
                bulk_update(all_data, readout_data, stats_data, rec_data,
                        fileinfodict)
                label = get_EHxADy_string(all_data)
                if label in prompt_like_events.keys():
                    prompt_like_events[label].append((i, all_data))
                else:
                    prompt_like_events[label] = [(i, all_data)]
            if (len(delayed_like_events) < max_delayeds_desired and
                    is_delayed_like(readout_data, stats_data, rec_data)):
                all_data = {}
                readout_data.unlazyconstruct()
                stats_data.unlazyconstruct()
                rec_data.unlazyconstruct()
                bulk_update(all_data, readout_data, stats_data, rec_data,
                        fileinfodict)
                label = get_EHxADy_string(all_data)
                if label in delayed_like_events.keys():
                    delayed_like_events[label].append((i, all_data))
                else:
                    delayed_like_events[label] = [(i, all_data)]
    return (prompt_like_events, delayed_like_events)

def setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('-o', '--output', help='name of output h5 file')
    parser.add_argument('--infiles',
            help='a text file listing all of the files to draw events from')
    parser.add_argument('--max-prompt', type=int, default=-1,
            help='max number of prompt events to read (-1 = all)')
    parser.add_argument('--max-delayed', type=int, default=-1,
            help='max number of delayed events to read (-1 = all)')
    parser.add_argument('-d', '--debug', action='store_true')
    return parser


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

    prompts_to_fetch = args.max_prompt
    delayeds_to_fetch = args.max_delayed
    all_prompts = {}
    all_delayeds = {}
    for rootfile in rootfilenames:
        file_prompts, file_delayeds = \
            get_prompt_delayed_like_data_for_file(rootfile,
                    prompts_to_fetch, delayeds_to_fetch)
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
        dset_to_save[i, :] = prepareEventDataForH5(prompt, delayed, dt)
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
    for i, name in enumerate(METADATA_NAMES):
        outdset.attrs[str(i)] = name
        outdset.attrs[name] = i

    h5file.close()
    print "Exiting..."
