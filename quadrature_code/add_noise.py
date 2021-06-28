"""
add_noise.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 03-11-2021

Description: Script to add simulated detector noise to pre-calculated images.

"""

from noise_maker import Noise_Maker

base_name = 'testset'

params = {

        ### Loading ###
        'load_dir_base':        './Data',
        'base_name':            base_name,
        ### Saving ###
        'save_dir_base':        './Noisy_Data',
        'do_save':              True,
        ### Observation ###
        'multi_SNRs':           [1,3,5,8,10],

}

#Load instance of noise maker
maker = Noise_Maker(params)

#Run script to add noise
maker.run_multiple_snr_script()
