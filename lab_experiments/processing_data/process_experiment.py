"""
process_experiment.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 06-28-2021

Description: Process experimental images.

"""

import numpy as np
from experiment_image_processor import Experiment_Image_Processor

params = {
    'all_runs':     ['data_1s_bin1'],
    'session':      'run__6_01_21',
    'is_med':       True,
    'mask_type':    'spiders',
}

for mask in ['spiders', 'none', 'round'][:2]:
    
    params['mask_type'] = mask
    proc = Experiment_Image_Processor(params)
    proc.run_script()
