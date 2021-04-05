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
save_dir = r'.\new_data'
record_name = 'record'

#Spacing between position steps [m]
dstep = width / nsteps

#Radius of random perturbations [m]
rad = np.sqrt(2) / 2 * dstep

#Build steps
steps = np.linspace(-width/2, width/2, num=nsteps)

#Imager parameters
params = {

    ### FLYER params ###
    'is_sim':               False,
    'save_dir':             save_dir,
    'do_save':              True,
    'verbose':              True,
    ### Camera params ###
    'exp_time':             30,
    'num_scans':            3,
    'camera_wait_temp':     True,
    'camera_stable_temp':   True,
    'camera_temp':          -70,
    'camera_pupil_frame':   [500,700,300,500],
    ### Motion params ###
    'zero_pos':             [6000,3000],      #[motor steps]

}

#pupilcenter
pup_cen = (565, 476)

#Load imager
imgur = Imager(params)

#Setup run
imgur.setup_run()

#Write signal of new image
with open(f'{save_dir}/{record_name}.csv', 'a') as f:
    f.write('#######################\n')
    f.write('#'+str(datetime.utcnow())+'\n')
    f.write('#'+'Pupil Center '+pup_cen+'\n')
    f.write('#######################\n')

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
