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
        'do_save':          True,
        'run':              run,
        'session':          session,
        'is_med':           True,
        'save_dir':         'Processed_Images',

    }

    proc = Experiment_Image_Processor(params)
    proc.run_script()
