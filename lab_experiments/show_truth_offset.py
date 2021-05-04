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

session = 'data_30s_bin1'
is_med = True
data_dir = './Results'
true_dir = './Truth_Results'
sensing_method = ['centroid', 'model'][0]

#Load quoted positions
dname = f'{data_dir}/{session}__none' + ['','__median'][int(is_med)] + '.h5'
with h5py.File(dname, 'r') as f:
    pos = f['positions'][()]

#Get truth positions
tname = f'{true_dir}/{session}__none' + ['','__median'][int(is_med)] + \
    f'__truth_{sensing_method}' + '.h5'

with h5py.File(tname, 'r') as f:
    tru = f['truth'][()]
    flg = f['flags'][()]

#Difference
diff = tru - pos

#Plot
plt.figure()
plt.plot(*diff.T, 'x')
plt.axhline(0, color='k', linestyle=':')
plt.axvline(0, color='k', linestyle=':')
breakpoint()
