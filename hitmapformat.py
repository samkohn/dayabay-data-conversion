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
        'dt_last_ad_muon',
        'dt_last_ad_shower_muon',
        'dt_last_wp_muon'
        )
NPIXELS = 192
NCHANNELS = 4
NMETADATA = len(METADATA_NAMES)
ENTRYSIZE = NMETADATA + NCHANNELS * NPIXELS

def getFlattenedData(event):
    event['nHitsAD'] = event['nHitsAD_prompt']
    event['chargeAD'] = event['chargeAD_prompt']
    event['timeAD'] = event['timeAD_prompt']
    event['ring'] = event['ring_prompt']
    event['column'] = event['column_prompt']
    charge2d_prompt, time2d_prompt = roottools.getChargesTime(event)
    event['nHitsAD'] = event['nHitsAD_delayed']
    event['chargeAD'] = event['chargeAD_delayed']
    event['timeAD'] = event['timeAD_delayed']
    event['ring'] = event['ring_delayed']
    event['column'] = event['column_delayed']
    charge2d_delayed, time2d_delayed = roottools.getChargesTime(event)
    # Reshape everything into one long vector
    flatteneds = [
        # reshape(-1) produces a 1D array
        charge2d_prompt.reshape(-1),
        time2d_prompt.reshape(-1),
        charge2d_delayed.reshape(-1),
        time2d_delayed.reshape(-1),
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
