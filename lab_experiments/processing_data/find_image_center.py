"""
find_image_center.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 06-30-2021

Description: Script to find the center of all images.

"""

import numpy as np
import h5py
import matplotlib.pyplot as plt;plt.ion()
import atexit

run = 'run__6_01_21__data_1s_bin1__none__median'
data_dir = './Arda_Results'

f = h5py.File(f'{data_dir}/{run}.h5', 'r')
atexit.register(f.close)
imgs = f['images']
poss = f['positions'][()]

cen = np.array([563,473])
cen0 = np.array([567,482])

ind0 = np.argmin(np.hypot(*poss.T))

ceni = np.array(imgs.shape[1:])/2

plt.imshow(imgs[ind0])
plt.plot(*ceni, 'r+')
plt.plot(*(ceni - (cen - cen0)), 'kx')
breakpoint()
