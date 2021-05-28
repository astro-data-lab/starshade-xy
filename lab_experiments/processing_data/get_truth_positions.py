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

all_runs = ['data_1s_bin1', 'data_1s_bin2', 'data_1s_bin4']
session = 'run__5_26_21'

is_med = False
data_dir = './Results'
true_dir = './Truth_Results'

for run in all_runs:

    fname = f'{data_dir}/{session}__{run}__none' + ['','__median'][int(is_med)] + '.h5'

    params = {
        'image_file':       fname,
        'debug':            [False, True][1],
        'do_save':          [False, True][0],
        'save_dir':         true_dir,
        'sensing_method':   ['centroid', 'model'][1],
    }

    sen = Other_Sensor(params)
    sen.get_positions()
