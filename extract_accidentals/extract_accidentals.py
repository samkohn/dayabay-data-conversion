"""
This script will create a sample of accidental IBD pairs (uncorrelated
background) that is statistically similar to the true accidental background
that contaminates the IBD sample.

"""
import roottools as rt
import matplotlib.pyplot as plt
import h5py
import numpy as np
import pickle
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

if __name__ == "__main__":
    outfilename = "accidentals.h5"
    h5file = h5py.File(outfilename, "w")
    rootfilename = ("/global/project/projectdirs/dayabay/data/exp/" +
        "dayabay/2011/p14a/Neutrino/1224/recon.Neutrino.0021221." +
        "Physics.EH1-Merged.P14A-P._0012.root")

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
        if is_IBD_trigger(readout_data):
            if is_prompt_like(readout_data, stats_data, rec_data):
                prompt_like_events.append(readout_data)
            if is_delayed_like(readout_data, stats_data, rec_data):
                delayed_like_events.append(readout_data)
        if i % 10000 == 0:
            logging.info("At entry number %d", i)
        if i == 20000:
            break
