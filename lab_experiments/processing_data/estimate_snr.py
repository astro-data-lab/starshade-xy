"""
estimate_snr.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 07-01-2021

Description: Script to estimate the SNR of experimental images.

"""

import numpy as np
import matplotlib.pyplot as plt;plt.ion()
import h5py
from astropy.io import fits
import image_util

run = 'data_1s_bin1'
# session = 'run__6_01_21'
session = 'run__5_26_21'
# session = 'run__8_30_21'

pos0 = [0,0]

is_med = [False, True][0]

############################################
####    Load Lab Data ####
############################################

data_dir = f'/home/aharness/Research/Frick_Lab/Data/FFNN/{session}/{run}'
record = np.genfromtxt(f'{data_dir}/record.csv', delimiter=',')

ind0 = np.argmin(np.hypot(record[:,1] - pos0[0], record[:,2] - pos0[0]))
inum = int(record[:,0][ind0])

with fits.open(f'{data_dir}/image__{str(inum).zfill(4)}.fits') as hdu:
    img = hdu[0].data.astype(float)
    texp = hdu[0].header['EXPOSURE']

if is_med:
    nframe = img.shape[0]
    img = np.median(img, 0)
else:
    nframe = 1
    img = img[0]

#Subtract background
back = np.median(np.concatenate((img[:10].flatten(), img[-10:].flatten(),
    img[:,:10].flatten(), img[:,-10:].flatten())))

img -= back

############################################
####    Get SNR ####
############################################

#Detector params
fwhm = 28
ccd_read = 4.78
ccd_gain = 0.768
ccd_dark = 7e-4
ccd_cic = 0.0025

#Center on peak
cen = np.unravel_index(np.argmax(img), img.shape)
cen = image_util.get_centroid_pos(img, cen[::-1], 2*fwhm)[::-1]

#Get radius of image around peak
rr = np.hypot(*(np.indices(img.shape).T - cen).T)

#Get total signal in FWHM
in_spot = rr <= fwhm/2
signal = img[in_spot].sum() * ccd_gain
num_ap = np.count_nonzero(in_spot)

#Get noise
noise = np.sqrt(signal + num_ap*(ccd_dark*texp + ccd_read**2. + ccd_cic))

#Compare SNR (mean per pixel)
snr = signal / noise / np.sqrt(num_ap)

print(f'\nSNR: {snr:.2f}\n')

noise2 = img[in_spot].std()*ccd_gain * np.sqrt(num_ap)
# noise2 = np.sqrt(noise2**2 - (img[in_spot].mean()*ccd_gain/5.5)**2)

snr2 = signal / noise2 / np.sqrt(num_ap)
print(f'\nSNR2: {snr2:.2f}\n')


plt.imshow(img)
breakpoint()
