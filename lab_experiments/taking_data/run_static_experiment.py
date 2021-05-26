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
    'num_scans':            50,
    'camera_wait_temp':     True,
    'camera_stable_temp':   True,
    'camera_temp':          -70,
    'camera_pupil_center':  (566, 476),
    'camera_pupil_width':   250,
    ### Motion params ###
    'zero_pos':             [-5500, 1500],      #[motor steps]

}

#Load imager
imgur = Imager(params)

#Setup run
imgur.setup_run()

#Move to center
imgur.move_to_pos(0,0)

#Take picture
imgur.take_picture(r'stable_30s')
