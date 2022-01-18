"""
add_noise.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 03-11-2021

Description: Script to add simulated detector noise to pre-calculated images.

"""

from noise_maker import Noise_Maker
import numpy as np

snrs = np.concatenate((np.arange(0.05, 1, 0.125), [3,5]))

load_dir = 'Newest_Data'
save_dir = load_dir + '_Noisy'

for base_name in ['trainset', 'testset']:

    print(f'\nRunning {base_name} ...')

    params = {

            ### Loading ###
            'load_dir_base':        load_dir,
            'base_name':            base_name,
            ### Saving ###
            'save_dir_base':        save_dir,
            'do_save':              True,
            ### Observation ###
            'count_rate':           7,
            'multi_SNRs':           snrs,
            ### Numerics ###
            'seed':                 888,
    }

    #Load instance of noise maker
    maker = Noise_Maker(params)

    #Run script to add noise
    maker.run_multiple_snr_script()
