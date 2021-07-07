"""
get_truth_positions.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 05-04-2021

Description: Script to estimate the true position of experimental images using
    other sensing scheme such as centroiding or non-linear least squares fit to a model.

"""

import numpy as np
from truth_sensor import Truth_Sensor

all_runs = ['data_1s_bin1', 'data_2s_bin1'][:1]
session = 'run__6_01_21'

is_med = True
data_dir = './Test_Results'
true_dir = './Test_Truth_Results'

for run in all_runs:

    fname = f'{data_dir}/{session}__{run}__none' + ['','__median'][int(is_med)] + '.h5'

    params = {
        'image_file':       fname,
        'debug':          True,
        'save_dir':         true_dir,
        'sensing_method':   ['centroid', 'model'][1],
    }

    sen = Truth_Sensor(params)
    sen.get_positions()
