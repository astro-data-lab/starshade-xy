"""
process_experiment.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 06-28-2021

Description: Process experimental images.

"""

import numpy as np
from experiment_image_processor import Experiment_Image_Processor

# runs = ['data_1s_bin2', 'data_d25s_bin4', 'data_d2s_bin3', 'data_d3s_bin3']
# session = 'run__8_24_21'

runs = ['data_5s_bin1']
session = 'run__11_15_21_b'

for run in runs:

    print(f'\nProcessing Run: {run} ...')

    params = {
        'data_dir':         '/home/aharness/Research/Frick_Lab/Data/FFNN',
        'session':          session,
        'run':              run,
        'cal_ext':          run.split('_')[-1],
        'is_median':        True,
        'save_dir':         'Processed_Images',
        'do_save':          True,
        # 'image_center': (116//2-5, 116//2),
        # 'do_plot':True,
    }

    proc = Experiment_Image_Processor(params)
    proc.run_script()
