"""
This script will create a sample of accidental IBD pairs (uncorrelated
background) that is statistically similar to the true accidental background
that contaminates the IBD sample.

"""
import roottools as rt
from hitmapformat import *
import matplotlib.pyplot as plt
import h5py
import numpy as np
import itertools
import logging
logging.basicConfig(level=logging.INFO)

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
    Qtot = stats['NominalCharge']
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

def is_singles_like(stats_data):
    """
        Return True if the event data suggests it is singles-like based only on
        other nearby IBD-like triggers.

        This is in essence a multiplicity cut:

          - last AD trigger and next AD trigger are both more than 400 us away

    """
    exclusion_time = 400e-3
    dts = {}
    dts['last_1'] = stats_data['dtLastAD1_ms']
    dts['last_2'] = stats_data['dtLastAD2_ms']
    dts['last_3'] = stats_data['dtLastAD3_ms']
    dts['last_4'] = stats_data['dtLastAD4_ms']
    dts['next_1'] = stats_data['dtNextAD1_ms']
    dts['next_2'] = stats_data['dtNextAD2_ms']
    dts['next_3'] = stats_data['dtNextAD3_ms']
    dts['next_4'] = stats_data['dtNextAD4_ms']

    # Some of the ADs are always set to -1 because they don't exist
    # This may mistakenly not veto the 0th event in each file which also has
    # ADs set to -1 since there is no previous trigger. I don't think that
    # matters.
    dts['last_1'] = dts['last_1'] if dts['last_1'] > 0 else time_shield + 1
    dts['last_2'] = dts['last_2'] if dts['last_2'] > 0 else time_shield + 1
    dts['last_3'] = dts['last_3'] if dts['last_3'] > 0 else time_shield + 1
    dts['last_4'] = dts['last_4'] if dts['last_4'] > 0 else time_shield + 1
    dts['next_1'] = dts['next_1'] if dts['next_1'] > 0 else time_shield + 1
    dts['next_2'] = dts['next_2'] if dts['next_2'] > 0 else time_shield + 1
    dts['next_3'] = dts['next_3'] if dts['next_3'] > 0 else time_shield + 1
    dts['next_4'] = dts['next_4'] if dts['next_4'] > 0 else time_shield + 1

    # Ensure that all of the dts are larger than the exclusion time
    return all(map(lambda dt: dt > exclusion_time, dts.values()))

def prepareEventDataForH5(prompt, delayed):
    """
        Return a numpy array with the relevant data for saving to HDF5.

        This method makes use of the hitmapformat.py module.
    """
    pass

def bulk_update(first, *args):
    """
        An extension of the dict.update which updates one dict with many more
        dicts.

        Like the normal dict.update, returns None.

    """
    for other in args:
        first.update(other)

    return None

if __name__ == "__main__":
    outfilename = "accidentals.h5"
    h5file = h5py.File(outfilename, "w")
    runno = '0021221'
    fileno = '0012'
    rootfilename = ("/global/project/projectdirs/dayabay/data/exp/" +
        "dayabay/2011/p14a/Neutrino/1224/recon.Neutrino.%s." +
        "Physics.EH1-Merged.P14A-P.%s.root") % (runno, fileno)

    readout = rt.makeCalibReadoutTree(rootfilename)
    stats = rt.makeCalibStatsTree(rootfilename)
    rec = rt.makeRecTree(rootfilename)

    num_triggers = readout.numEntries()
    assert num_triggers == stats.numEntries(), "len(readout) != len(stats)"
    assert num_triggers == rec.numEntries(), "len(readout) != len(rec)"
    logging.info("num triggers = %d", num_triggers)
    # Determine which events are prompt-like and which ones are delayed-like
    # (allow for overlap: in particular, all delayed-like events are also
    # prompt-like).
    prompt_like_events = []
    delayed_like_events = []
    for i, (readout_data, stats_data, rec_data) in enumerate(itertools.izip(readout.getentries(),
            stats.getentries(), rec.getentries())):
        logging.debug('i = %d', i)
        if is_IBD_trigger(readout_data) and is_singles_like(stats_data):
            if is_prompt_like(readout_data, stats_data, rec_data):
                all_data = {}
                bulk_update(all_data, readout_data, stats_data, rec_data)
                prompt_like_events.append((i, all_data))
            if is_delayed_like(readout_data, stats_data, rec_data):
                all_data = {}
                bulk_update(all_data, readout_data, stats_data, rec_data)
                prompt_like_events.append((i, all_data))
        if len(prompt_like_events > 10):
            break

    # Now the two lists contain the data needed to assemble a set of
    # accidentals.

    # Algorithm: Shuffle the two lists' orders. Then pair up events. As a
    # trivial safety measure, ensure that no event is paired up with itself.
    # TODO: this algorithm mixes different ADs together. Fix that.
    prompt_like_events = np.array(prompt_like_events)
    delayed_like_events = np.array(delayed_like_events)
    np.random.shuffle(prompt_like_events)
    np.random.shuffle(delayed_like_events)
    # Note: python's zip method discards any unpaired events. This is fine in
    # our case since there's no default and we don't want repeats.
    pairs = zip(prompt_like_events, delayed_like_events)

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

    for dt, (prompt, delayed) in zip(dts, pairs):
        event = {}
        event['runno'] = runno  # from original file IO
        event['fileno'] = fileno  # from original file IO
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
        #TODO
