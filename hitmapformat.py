'''This module handles the hitmap "format" in terms of the array storage, e.g. to
convert from 8x24 to 1x192, to store metadata, etc.'''

import numpy as np
import roottools

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
        'dt',
        'dt_last_ad_muon',
        'dt_last_ad_shower_muon',
        'dt_last_wp_muon'
        )
NPIXELS = 192
NCHANNELS = 4
NMETADATA = len(METADATA_NAMES)
ENTRYSIZE = NMETADATA + NCHANNELS * NPIXELS

def getFlattenedData(charge_prompt, time_prompt, charge_delayed, time_delayed):
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
    """Expect a 779-length 1D numpy array or similar. Split it into four 8x24
    images plus the metadata, and return a dict with the appropriate
    attributes.
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
