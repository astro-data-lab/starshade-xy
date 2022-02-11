"""
process_experiment.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 06-28-2021

Description: Process experimental images.

"""

import numpy as np
from experiment_image_processor import Experiment_Image_Processor


# session = 'run__6_01_21'
# runs = ['data_1s_bin1', 'data_2s_bin1']
# is_flyer = False

# session = 'run__8_30_21'
# runs = ['data_1s_bin1']
# is_flyer = False

session = 'CNN__12_20_21'
# session = 'CNN__12_21_21'
# session = 'CNN__12_21_21_b'
runs = ['t15_wide']
is_flyer = True

if is_flyer:
    data_dir = '/home/aharness/Research/Formation_Flying/Flyer_Results'
else:
    data_dir = '/home/aharness/Research/Frick_Lab/Data/FFNN'

for run in runs:

    print(f'\nProcessing Session: {session}, Run: {run} ...\n')

    params = {
        'data_dir':         data_dir,
        'session':          session,
        'run':              run,
        'is_median':        True,
        'save_dir':         'Processed_Images',
        'do_save':          True,
        'is_flyer_data':    is_flyer,
    }

    proc = Experiment_Image_Processor(params)
    proc.run_script()
