"""
run_experiment.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 04-02-2021

Description: Script to take a a number of pupil plane images in the Princeton
    starshade testbed stepping across a grid of constant x,y spacing.

"""

import numpy as np
from imager import Imager
from datetime import datetime

#Number of steps
nsteps = 20

#Width of motion grid [m]
width = 1.5e-3

#Saving
save_dir = r'C:\Users\starshade_lab\repos\starshade-xy\lab_experiments\new'
record_name = 'record'

#Spacing between position steps [m]
dstep = width / nsteps

#Radius of random perturbations [m]
rad = np.sqrt(2) / 2 * dstep

#Build steps
steps = np.linspace(-width/2, width/2, num=nsteps)

#pupilcenter
pup_cen = (567, 475)

#Imager parameters
params = {

    ### FLYER params ###
    'is_sim':               False,
    'save_dir':             save_dir,
    'do_save':              True,
    'verbose':              True,
    ### Camera params ###
    'exp_time':             60,
    'num_scans':            3,
    'camera_wait_temp':     True,
    'camera_stable_temp':   True,
    'camera_temp':          -70,
    'camera_pupil_center':  pup_cen,
    'camera_pupil_width':   250,
    'binning':              4,
    ### Motion params ###
    'zero_pos':             [2500, 7000],      #[motor steps]

}

#Write signal of new image
with open(f'{save_dir}/{record_name}.csv', 'a') as f:
    f.write('#######################\n')
    f.write('#'+str(datetime.utcnow())+'\n')
    f.write(f'# Pupil Center: {pup_cen[0]:.0f}, {pup_cen[1]:.0f}\n')
    f.write('#######################\n')

#Load imager
imgur = Imager(params)

#Setup run
imgur.setup_run()

#Loop over positions and take images
i = 1
for x in steps:
    print(f'Running x step # {i // len(steps) + 1}')
    for y in steps:

        #Get shift of telescope
        cx = x + 2 * rad * (np.random.random_sample() - 0.5)
        cy = y + 2 * rad * (np.random.random_sample() - 0.5)

        #Move camera
        imgur.move_to_pos(cx, cy)

        #Take image
        imgur.take_picture(i)

        #Record exact position
        with open(f'{save_dir}/{record_name}.csv', 'a') as f:
            f.write(f'{str(i).zfill(4)}, {cx}, {cy}\n')

        #Increment
        i += 1