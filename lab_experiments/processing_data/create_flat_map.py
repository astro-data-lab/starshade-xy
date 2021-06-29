"""
compare_model_lab.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 06-29-2021

Description: Script to create flat-field map of pupil image.

"""

import numpy as np
import matplotlib.pyplot as plt;plt.ion()
from astropy.io import fits
import glob
import h5py
import image_util

data_dir = '/home/aharness/Research/Frick_Lab/Data/FFNN/pupil_flats'

#Grab filenames
img_nums = np.arange(1, 7)

#Grab data
imgs = []
for i in img_nums:
    with fits.open(f'{data_dir}/img_{i}.fits') as hdu:
        imgs.extend(hdu[0].data)

imgs = np.array(imgs)

#Get median image
med = np.median(imgs, 0)

#Crop
# cen = (563, 473)
cen = (553, 467)
wid = 116
med = image_util.crop_image(med, cen, wid//2)

#Normalize
med /= med.max()

plt.imshow(med)

if [False, True][0]:
    with h5py.File('./flat.h5', 'w') as f:
        f.create_dataset('data', data=med)

breakpoint()
