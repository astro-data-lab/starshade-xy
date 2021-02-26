"""
show_convergence.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 02-26-2021

Description: Script to plot the results of the convergence of the number of
    occulter points per starshade edge.

"""

import numpy as np
import h5py
import matplotlib.pyplot as plt

#Load data
with h5py.File('./xtras/convergence_results.h5', 'r') as f:
    emaps = f['emaps'][()]
    pts = f['num_occ_pts'][()]

#Calculate maximum relative change from final run
diff = np.abs(emaps - emaps[-1]).max((1,2)) / np.abs(emaps[-1]).max()

#Plot
plt.semilogy(pts, diff, 'o')
plt.xlabel('Number of occulter points per edge (num_occ_pts)')
plt.ylabel('Relative Difference from final run')
plt.show()
