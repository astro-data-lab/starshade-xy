"""
add_aperture.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 04-05-2021

Description: Script to add spiders and secondary to experimental images.

"""

import numpy as np
import matplotlib.pyplot as plt;plt.ion()
from astropy.io import fits
import h5py
import glob
from scipy.ndimage import affine_transform

#Choose aperture size for proper scaling to space
num_pts = 96

#aperture size (from pupil magnification)
Dtel = num_pts * 1.748*13e-6

#TODO: hide
data_dir = '/home/aharness/Research/Frick_Lab/Data/FFNN/data'

#Read record
record = np.genfromtxt(f'{data_dir}/record.csv', delimiter=',')

def get_image(fname):
    with fits.open(fname) as hdu:
        data = hdu[0].data
        utmp = hdu[0].header['UNSTTEMP']
    return data, utmp

#Get background image
with fits.open(f'{data_dir}/background.fits') as hdu:
    back = np.median(hdu[0].data, 0)

#Get image shape
img_shp = back.shape

#Loop through steps and get images
imgs = np.empty((0,) + img_shp)
locs = np.empty((0, 2))
for i in range(len(record))[450:]:

    #Current step number
    stp = str(int(record[i][0])).zfill(4)

    #Current position
    pos = record[i][1:]

    #Get image
    img, utmp = get_image(f'{data_dir}/image__{stp}.fits')
    print(np.median(img), utmp)
    breakpoint()

    #Store images + positions
    imgs = np.concatenate((imgs, img))
    locs = np.concatenate((locs, [pos]*img.shape[0]))

#Subtract background
# imgs -= back

#Load Pupil Mask
with h5py.File(f'../diffraction_code/xtras/pupil_mask.h5', 'r') as f:
    full_mask = f['mask'][()]

#Do affine transform
scaling = full_mask.shape[0]/num_pts
dx = -scaling*(img_shp[0]-num_pts)/2
affmat = np.array([[scaling, 0, dx], [0, scaling, dx]])
pupil_mask = affine_transform(full_mask, affmat, output_shape=img_shp, order=5)

#Add bounds
pupil_mask[pupil_mask < 0] = 0
pupil_mask[pupil_mask > 1] = 1

#Apply  mask

plt.imshow(imgs[0])
breakpoint()
