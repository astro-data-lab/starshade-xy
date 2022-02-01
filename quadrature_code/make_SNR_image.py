"""
make_SNR_image.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 01-18-2022

Description: Script to generate a centered, unblocked image to use for SNR tests.

"""

import numpy as np
import diffraq
import os

#User options
apod_name = 'm12p8'
wave = 403e-9
num_tel_pts = 96
#Telescope sizes in Lab and Space coordinates [m] (sets scaling factor)
Dtel_lab = 2.201472e-3

############################

#Specify simulation parameters
params = {
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
sim = diffraq.Simulator(params)
sim.setup_sim()

#Get diffraction and convert to intensity
img = np.abs(sim.calculate_diffraction())**2

#Save image
np.save('./xtras/snr_image.npy', img)
