"""
generate_images.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 02-26-2021

Description: Script to generate a number of pupil plane images stepping across
    a grid of constant x,y spacing.

"""

import numpy as np
from bdw import BDW

#Name of data file to save
save_name = 'stepped_data'

#Width of motion grid [m]
width = 2.5e-3

#Spacing between position steps [m]
dstep = width / 32

#Radius of random perturbations [m]
rad = np.sqrt(2) / 2 * dstep

#Specify simulation parameters
params = {
    ### Lab ###
    'wave':             0.405e-6,       #Wavelength of light [m]

    ### Telescope ###
    'tel_diameter':     5e-3,           #Telescope aperture diameter [m]
    'num_tel_pts':      54,             #Size of grid to calculate over pupil
    'with_spiders':     True,           #Superimpose spiders on pupil image?

    ### Starshade ###
    'apod_name':        'lab_ss',       #Apodization profile name. Options: ['lab_ss', 'circle']
    'num_petals':       12,             #Number of starshade petals

    ### Saving ###
    'do_save':          False,          #Save data?
    'verbose':          False,          #Print out details?
}

#Load BDW class
bdw = BDW(params)

#Build steps
steps = np.arange(-width/2, width/2 + dstep, dstep)

#Containers
images = np.empty((0, bdw.num_pts, bdw.num_pts))
positions = np.empty((0, 2))

#Loop over steps in each axis and calculate image
i = 1
for x in steps:
    print(f'Running x step # {i // len(steps) + 1}')
    for y in steps:

        #Set shift of telescope
        x += 2 * rad * (np.random.random_sample() - 0.5)
        y += 2 * rad * (np.random.random_sample() - 0.5)
        bdw.tel_shift = [x, y]

        #Get diffraction and convert to intensity
        img = np.abs(bdw.calculate_diffraction())**2

        #Save and write position to csv
        np.save(f'test/{str(i).zfill(4)}', img)
        with open('test.csv', 'a') as f:
            f.write(f'{i}, {x}, {y}\n')
        i += 1
