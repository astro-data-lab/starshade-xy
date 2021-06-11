"""
generate_images.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 02-26-2021

Description: Script to generate a number of pupil plane images stepping across
    a grid of constant x,y spacing.

"""

import numpy as np
import diffraq

#Name of data file to save
save_name = 'stepped_data'

#Number of steps
nsteps = 48

#Width of motion grid [m]
width = 2.5e-3

#Spacing between position steps [m]
dstep = width / nsteps

#Radius of random perturbations [m]
rad = np.sqrt(2) / 2 * dstep

#Specify simulation parameters
params = {
    ### Lab ###
    'wave':             0.405e-6,       #Wavelength of light [m]

    ### Telescope ###
    'tel_diameter':     2.2e-3,         #Telescope aperture diameter [m]
    'num_tel_pts':      96,             #Size of grid to calculate over pupil
    'with_spiders':     True,           #Superimpose spiders on pupil image?

    ### Starshade ###
    'apod_name':        'lab_ss',       #Apodization profile name. Options: ['lab_ss', 'circle']
    'num_petals':       12,             #Number of starshade petals

    ### Saving ###
    'do_save':          False,          #Save data?
    'verbose':          False,          #Print out details?
}

#Load simulator
sim = diffraq.Simulator(params)

#Build steps
steps = np.linspace(-width/2, width/2, num=nsteps)

#Containers
images = np.empty((0, sim.num_pts, sim.num_pts))
positions = np.empty((0, 2))

#Loop over steps in each axis and calculate image
i = 1
for x in steps:
    print(f'Running x step # {i // len(steps) + 1}')
    for y in steps:

        #Set shift of telescope
        nx = x + 2 * rad * (np.random.random_sample() - 0.5)
        ny = y + 2 * rad * (np.random.random_sample() - 0.5)
        sim.tel_shift = [nx, ny]

        #Get diffraction and convert to intensity
        img = np.abs(sim.calculate_diffraction())**2

        #Save and write position to csv
        np.save(f'test/{str(i).zfill(4)}', img)
        with open('test.csv', 'a') as f:
            f.write(f'{str(i).zfill(4)}, {nx}, {ny}\n')
        i += 1
