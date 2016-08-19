'''This module handles the hitmap "format" in terms of the array storage, e.g. to
convert from 8x24 to 1x192, to store metadata, etc.'''

import numpy as np

__all__ = [
        'METADATA_NAMES',
        'NPIXELS',
        'NCHANNELS',
        'NMETADATA',
        'ENTRYSIZE',
        'getFlattenedData',
        'unflattenData'
        ]

METADATA_NAMES = (
        'runno',
        'fileno',
        'site',
        'det',
        'time_sec',
        'time_nanosec',
        'trigno_prompt',
        'trigno_delayed',
        'dt_IBD',
        'dt_last_ad_muon',
        'dt_last_ad_shower_muon',
        'dt_last_wp_muon'
        )
NPIXELS = 192
NCHANNELS = 4
NMETADATA = len(METADATA_NAMES)
ENTRYSIZE = NMETADATA + NCHANNELS * NPIXELS

def getFlattenedData(event, charge_prompt, time_prompt, charge_delayed, time_delayed):
    """Given metadata stored in "event," as well as the hitmaps in charge and
    time for prompt and delayed triggers, reformat the data into a 1D numpy
    array so it can be stored more easily."""
    # Reshape everything into one long vector
    flatteneds = [
        # reshape(-1) produces a 1D array
        charge_prompt.reshape(-1),
        time_prompt.reshape(-1),
        charge_delayed.reshape(-1),
        time_delayed.reshape(-1),
        np.array([event[name] for name in METADATA_NAMES])
    ]
    return np.hstack(flatteneds)

def unflattenData(datavec):
    """Expect an ENTRYSIZE-length 1D numpy array or similar. Split it into four 8x24
    images plus the metadata, and return a dict with the
    appropriately-formatted data.
    """
    event = {}
    shape = (8, 24)
    event['charge_prompt'] = datavec[0:NPIXELS].reshape(shape)
    event['time_prompt'] = datavec[NPIXELS:(2*NPIXELS)].reshape(shape)
    event['charge_delayed'] = datavec[(2*NPIXELS):(3*NPIXELS)].reshape(shape)
    event['time_delayed'] = datavec[(3*NPIXELS):(4*NPIXELS)].reshape(shape)
    for i, name in enumerate(METADATA_NAMES):
        event[name] = datavec[(4*NPIXELS+i)]
    return event
