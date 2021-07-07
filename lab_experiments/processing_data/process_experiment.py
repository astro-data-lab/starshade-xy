"""
process_experiment.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 06-28-2021

Description: Process experimental images.

"""

import numpy as np
from experiment_image_processor import Experiment_Image_Processor

runs = ['data_1s_bin1']
session = 'run__6_01_21'

for run in runs:

    print(f'\nProcessing Run: {run} ...')

    params = {
        'data_dir':         '/home/aharness/Research/Frick_Lab/Data/FFNN',
        'run':              run,
        'session':          session,
        'is_median':        True,
        'save_dir':         'Processed_Images',
        'do_save':          True,

    }

    proc = Experiment_Image_Processor(params)
    proc.run_script()
