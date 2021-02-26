"""
run_convergence.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 02-26-2021

Description: Script to run a number of BDW simulations to test the convergence
    of the number of occulter points per starshade edge.

"""

import numpy as np
from bdw import BDW
import h5py
import time

#Specify simulation parameters
params = {
    'num_tel_pts':       64,
    'apod_name':        'lab_ss',
    'do_save':          False,
    'verbose':          False,
}

#List of number of points to test
pts = range(1000, 11000, 1000)

#Loop through and calculate maps
emaps = []
for p in pts:

    print(f'Running num_occ_pts: {p}')
    tik = time.perf_counter()

    #Change parameters
    params['num_occ_pts'] = p
    params['save_ext'] = p

    #Load BDW class
    bdw = BDW(params)

    #Run simulation
    emaps.append( bdw.run_sim() )

    #Printout
    print(f'Time: {time.perf_counter()-tik:.2f} [s]')

#Save data
with h5py.File('./xtras/convergence_results.h5', 'w') as f:
    f.create_dataset('emaps', data=emaps)
    f.create_dataset('num_occ_pts', data=np.array(pts))
