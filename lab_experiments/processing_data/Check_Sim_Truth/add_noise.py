"""
add_noise.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 03-11-2021

Description: Script to add simulated detector noise to pre-calculated images.

"""


import numpy as np
import os
import imp

noise_maker = imp.load_source("noise_maker", os.path.join(os.pardir, os.pardir, \
    os.pardir, 'quadrature_code', 'noise_maker.py'))

snrs = [0.4]

load_dir = './Sml_Data'
save_dir = './Sml_Noisy_Data'

base_name = 'sim_check'

print(f'\nRunning {base_name} ...')

params = {

        ### Loading ###
        'load_dir_base':        load_dir,
        'base_name':            base_name,
        ### Saving ###
        'save_dir_base':        save_dir,
        'do_save':              True,
        ### Observation ###
        'count_rate':           200,
        'multi_SNRs':           snrs,

}

#Load instance of noise maker
maker = noise_maker.Noise_Maker(params)

#Run script to add noise
maker.run_multiple_snr_script()
