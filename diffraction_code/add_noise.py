"""
add_noise.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 03-11-2021

Description: Script to add simulated detector noise to pre-calculated images.

"""

from noise_maker import Noise_Maker

params = {
        ### Loading ###
        'load_dir':         '../data/train',            #Directory to load pre-made images
        ### Saving ###
        'save_dir':         './noisy_data/train',       #Directory to save images with noise added
        'do_save':          True,                       #Save data?
}

#Load instance of noise maker
maker = Noise_Maker(params)

#Run script to add noise
maker.run_script()
