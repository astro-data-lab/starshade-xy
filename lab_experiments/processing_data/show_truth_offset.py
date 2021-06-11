"""
show_truth_offset.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 05-04-2021

Description: Plot offset between quoted position and centroided position

"""

import numpy as np
import h5py
import matplotlib.pyplot as plt;plt.ion()

session = 'run__6_01_21'
run = 'data_1s_bin1'
is_med = True
data_dir = './Results'
true_dir = './Truth_Results'

#Load quoted positions
dname = f'{data_dir}/{session}__{run}__none' + ['','__median'][int(is_med)] + '.h5'
with h5py.File(dname, 'r') as f:
    pos = f['positions'][()]

#Get model truth positions
mname = f'{true_dir}/{session}__{run}__none' + ['','__median'][int(is_med)] + \
    f'__truth_model' + '.h5'

with h5py.File(mname, 'r') as f:
    mtru = f['truth'][()]
    mflg = f['flags'][()]

#Get centroid truth positions
cname = f'{true_dir}/{session}__{run}__none' + ['','__median'][int(is_med)] + \
    f'__truth_centroid' + '.h5'

with h5py.File(cname, 'r') as f:
    ctru = f['truth'][()]
    cflg = f['flags'][()]


#Difference
mdiff = mtru - pos
cdiff = ctru - pos

#Plot
plt.figure()
plt.plot(*mdiff.T, 'x', label='Model')
plt.plot(*cdiff.T, '+', label='Centroid')
plt.axhline(0, color='k', linestyle=':')
plt.axvline(0, color='k', linestyle=':')
plt.legend()
breakpoint()
