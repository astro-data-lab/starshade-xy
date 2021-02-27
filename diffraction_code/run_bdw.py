"""
run_sim.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 02-26-2021

Description: Script to run a BDW simulation.

"""

from bdw import BDW

#Specify simulation parameters
params = {
    ### Lab ###
    'wave':             0.405e-6,       #Wavelength of light [m]
    'z0':               27.5,           #Source - starshade distance [m]
    'z1':               50.,            #Starshade - telescope distance [m]

    ### Telescope ###
    'tel_diameter':     5e-3,           #Telescope aperture diameter [m]
    'num_tel_pts':      64,             #Size of grid to calculate over pupil
    'tel_shift':        [0, 0],         #(x,y) shift of telescope relative to starshade-source line [m]

    ### Starshade ###
    'apod_name':        'lab_ss',       #Apodization profile name. Options: ['lab_ss', 'circle']
    'with_spiders':     False,          #Superimpose secondary mirror spiders on pupil image?

    ### Saving ###
    'save_dir_base':    './',           #Base directory to save data
    'session':          '',             #Session name, i.e., subfolder to save data
    'save_ext':         '',             #Save extension to append to data
    'do_save':          True,           #Save data?
}

#Load BDW class
bdw = BDW(params)

#Run simulation
bdw.run_sim()
