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
import os

#Save directory
base_dir = 'Newest_Data'

#Width of motion grid in Space coordinates [m]
grid_width_space = 3.

#Number of steps
num_steps = {'testset':20, 'trainset':50}

#User options
apod_name = 'm12p8'
with_spiders = True
wave = 403e-9
num_tel_pts = 96
#Telescope sizes in Lab and Space coordinates [m] (sets scaling factor)
Dtel_lab = 2.201472e-3
Dtel_space = 2.4
#Random number generator seed
seed = 888

############################

#Get random number generatore
rng = np.random.default_rng(seed)

#Lab to space scaling
lab2space = Dtel_space / Dtel_lab
space2lab = 1/lab2space

#Lab width
grid_width_lab = grid_width_space * space2lab

#Loop over training and testing
for base_name in ['trainset', 'testset']:

    print(f'\nRunning {base_name} ...')

    #Create directory
    save_dir = os.path.join('Simulated_Images', base_dir, base_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    #Number of steps
    nsteps = num_steps[base_name]

    #Spacing between position steps [m]
    dstep = grid_width_lab / nsteps

    #Radius of random perturbations [m]
    rad = np.sqrt(2) / 2 * dstep

    #Specify simulation parameters
    params = {
        ### Lab ###
        'wave':             wave,                   #Wavelength of light [m]

        ### Telescope ###
        'tel_diameter':     Dtel_lab,               #Telescope aperture diameter [m]
        'num_tel_pts':      num_tel_pts,            #Size of grid to calculate over pupil

        ### Starshade ###
        #will specify apod_name after circle is run
        'num_petals':       12,                     #Number of starshade petals

        ### Saving ###
        'do_save':          False,                  #Don't save data
        'verbose':          False,                  #Silence output
    }

    #Run unblocked image first (lab calibration)
    params['apod_name'] = 'circle'
    params['circle_rad'] = 25.086e-3
    sim = diffraq.Simulator(params)
    sim.setup_sim()
    cal_img = np.abs(sim.calculate_diffraction())**2

    #Save image
    cal_file = os.path.join(save_dir, 'calibration')
    np.save(cal_file, cal_img)

    #New simulator for starshade images
    params['with_spiders'] = with_spiders
    params['apod_name'] = apod_name
    sim = diffraq.Simulator(params)
    sim.setup_sim()

    #Build steps
    steps = np.linspace(-grid_width_lab/2, grid_width_lab/2, num=nsteps)

    #Containers
    images = np.empty((0, sim.num_pts, sim.num_pts))
    positions = np.empty((0, 2))

    #Create new csv file
    csv_file = os.path.join(save_dir, base_name + '.csv')
    with open(csv_file, 'w') as f:
        #Write out header
        meta_hdr = '#,grid_width_space,apod_name,with_spiders,wave,num_tel_pts,Dtel_lab,Dtel_space,seed\n'
        meta_val = f'#,{grid_width_space},{apod_name},{with_spiders},{wave},{num_tel_pts},{Dtel_lab},{Dtel_space},{seed}\n'
        f.write(meta_hdr)
        f.write(meta_val)

    #Loop over steps in each axis and calculate image
    i = 0
    for x in steps:
        print(f'Running x step # {i // len(steps) + 1} / {len(steps)}')
        for y in steps:

            #Set shift of telescope
            nx = x + 2 * rad * (rng.random() - 0.5)
            ny = y + 2 * rad * (rng.random() - 0.5)
            sim.tel_shift = [nx, ny]

            #Get diffraction and convert to intensity
            img = np.abs(sim.calculate_diffraction())**2

            #Number string
            num_str = str(i).zfill(6)

            #Save image
            img_file = os.path.join(save_dir, num_str)
            np.save(img_file, img)

            #Write position to csv
            with open(csv_file, 'a') as f:
                f.write(f'{num_str}, {nx}, {ny}\n')
            i += 1
