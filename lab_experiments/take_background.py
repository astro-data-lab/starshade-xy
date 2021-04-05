"""
get_lab_exposure_time.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 04-02-2021

Description: Script to calculate lab exposure time needed to reach certain SNR.
"""

import numpy as np
from imager import Imager
import matplotlib.pyplot as plt;plt.ion()

exp_time = 30

##############

#Imager parameters
params = {

    ### FLYER params ###
    'is_sim':               False,
    'do_save':              True,
    'verbose':              True,
    ### Camera params ###
    'exp_time':             exp_time,
    'num_scans':            5,
    'camera_wait_temp':     True,
    'camera_stable_temp':   True,
    'camera_temp':          -70,
    'camera_pupil_frame':   [500,700,300,500],
    ### Motion params ###
    'zero_pos':             [6000, 3000],      #[motor steps]

}

#Load imager
imgur = Imager(params)

#Setup run
imgur.setup_run()

#Take picture
imgur.take_picture(r'background')
