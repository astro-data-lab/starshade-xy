"""
get_truth_positions.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 05-04-2021

Description: Script to estimate the true position of experimental images using
    other sensing scheme such as centroiding or non-linear least squares fit to a model.

"""

import numpy as np
from other_sensor import Other_Sensor

all_sessions = ['data_30s_bin1', 'data_30s_bin2', 'data_20s_bin4', \
    'data_60s_bin4']

is_med = True
data_dir = './Results'
true_dir = './Truth_Results'

for session in all_sessions:

    fname = f'{data_dir}/{session}__none' + ['','__median'][int(is_med)] + '.h5'

    params = {
        'image_file':       fname,
        'debug':            [False, True][0],
        'do_save':          [False, True][1],
        'save_dir':         true_dir,
        'sensing_method':   ['centroid', 'model'][0],
    }

    sen = Other_Sensor(params)
    sen.get_positions()
