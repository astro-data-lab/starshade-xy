"""
test_snr.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 01-18-2022

Description: Script to check Noise_Maker adds noise correctly to reach the target SNR.

"""

import numpy as np
import diffraq
from noise_maker import Noise_Maker

#SNRs to test
snrs = np.concatenate((np.arange(0.05, 1, 0.125), [3,5]))

#Image options
apod_name = 'm12p8'
wave = 403e-9
num_tel_pts = 96
#Telescope sizes in Lab and Space coordinates [m] (sets scaling factor)
Dtel_lab = 2.201472e-3

############################

#Specify simulation parameters
diffraq_params = {
    ### Lab ###
    'wave':             wave,                   #Wavelength of light [m]

    ### Telescope ###
    'tel_diameter':     Dtel_lab,               #Telescope aperture diameter [m]
    'num_tel_pts':      num_tel_pts,            #Size of grid to calculate over pupil
    'with_spiders':     False,

    ### Starshade ###
    'apod_name':        apod_name,
    'num_petals':       12,                     #Number of starshade petals

    ### Saving ###
    'do_save':          False,                  #Don't save data
    'verbose':          False,                  #Silence output
}

#Load simulator
sim = diffraq.Simulator(diffraq_params)
sim.setup_sim()

#Get diffraction and convert to intensity
img = np.abs(sim.calculate_diffraction())**2

############################

#Noise Maker params
noise_params = {
        'is_test':              True,
        'count_rate':           7,
        'multi_SNRs':           snrs,
}

#Load instance of noise maker
maker = Noise_Maker(noise_params)

#Test
maker.run_snr_test(img)
