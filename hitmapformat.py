'''This module handles the hitmap "format" in terms of the array storage, e.g. to
convert from 8x24 to 1x192, to store metadata, etc.'''

import numpy as np

__all__ = [
        'METADATA_NAMES',
        'NPIXELS',
        'NCHANNELS',
        'NMETADATA',
        'ENTRYSIZE',
        'VERSION',
        'getFlattenedData',
        'unflattenData',
        'label2id',
        'id2label',
        'labels'
        ]

METADATA_NAMES = {
        1: (
            'version',
            'runno',
            'fileno_prompt',
            'fileno_delayed',
            'site',
            'det',
            'time_sec',
            'time_nanosec',
            'trigno_prompt',
            'trigno_delayed',
            'energy_prompt',
            'energy_delayed',
            'dt_IBD',
            'dt_last_ad_muon',
            'dt_last_ad_shower_muon',
            'dt_last_wp_muon',
            'label_id'
            )
}
VERSION = 1
NPIXELS = 192
NCHANNELS = 4
NMETADATA = len(METADATA_NAMES[VERSION])
ENTRYSIZE = NMETADATA + NCHANNELS * NPIXELS

def getFlattenedData(event, charge_prompt, time_prompt, charge_delayed, time_delayed):
    """Given metadata stored in "event," as well as the hitmaps in charge and
    time for prompt and delayed triggers, reformat the data into a 1D numpy
    array so it can be stored more easily."""
    if charge_delayed is None:
        charge_delayed = np.zeros_like(charge_prompt)
    if time_delayed is None:
        time_delayed = np.zeros_like(time_prompt)
    # Fetch the id number of the event label
    label = event.get('label_id', 'unknown')
    event['label_id'] = label2id(label)
    event['version'] = VERSION
    # Reshape everything into one long vector
    flatteneds = [
        # reshape(-1) produces a 1D array
        charge_prompt.reshape(-1),
        time_prompt.reshape(-1),
        charge_delayed.reshape(-1),
        time_delayed.reshape(-1),
        np.array([event[name] for name in METADATA_NAMES[VERSION]])
    ]
    # reset the event['label_id'] entry
    event['label_id'] = label
    return np.hstack(flatteneds)

def unflattenData(datavec, version):
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
    for i, name in enumerate(METADATA_NAMES[version]):
        event[name] = datavec[(4*NPIXELS+i)]
    event['label_id'] = id2label(event.get('label_id', -1))
    return event

labels = {
        '-1': ('unknown', -1),
        'unknown': ('unknown', -1),
        '0': ('ibd', 0),
        'ibd': ('ibd', 0),
        '1': ('accidental', 1),
        'accidental': ('accidental', 1),
        '2': ('flasher', 2),
        'flasher': ('flasher', '2')
        }

def label2id(label):
    """Assign a unique id to this label."""
    return labels[label][1]

def id2label(idnum):
    """Return the label corresponding to this id."""
    return labels[str(int(idnum))][0]
