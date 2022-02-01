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

run = 'data_1s_bin1'
session = 'run__6_01_21'

pos0 = [0,0]

is_med = True

data_dir = './Arda_Results'

############################################
####    Load Lab Data ####
############################################

# #Load lab data
# lab_name = f'{session}__{run}__none{["", "__median"][int(is_med)]}'
# with h5py.File(f'{data_dir}/{lab_name}.h5', 'r') as f:
#
#     #get center
#     cen0 = np.array(pos0) * f['tel_diameter']/2
#
#     #Positions
#     tru_pos = f['positions'][()]
#
#     #Find image closest to specified point
#     ind0 = np.argmin(np.hypot(*(tru_pos - cen0).T))
#     pos = tru_pos[ind0]
#
#     #Get on-axis image
#     img = f['images'][ind0]
#
#     #Extras
#     texp = f['meta'][ind0][[0]]
#     cal_val = f['cal_value'][()]

data_dir = f'/home/aharness/Research/Frick_Lab/Data/FFNN/{session}/{run}'
record = np.genfromtxt(f'{data_dir}/record.csv', delimiter=',')

ind0 = np.argmin(np.hypot(record[:,1] - pos0[0], record[:,2] - pos0[0]))
inum = int(record[:,0][ind0])

with fits.open(f'{data_dir}/image__{str(inum).zfill(4)}.fits') as hdu:
    img = hdu[0].data.astype(float)
    texp = hdu[0].header['EXPOSURE']

if is_med:
    img = np.median(img, 0)
else:
    img = img[0]

############################################
####    Get SNR ####
############################################

#Detector params
fwhm = 28
ccd_read = 4.78
ccd_bias = 500
ccd_gain = 0.768
ccd_dark = 7e-4
ccd_cic = 0.0025

#Center on peak
cen = np.unravel_index(np.argmax(img), img.shape)

#Get radius of image around peak
rr = np.hypot(*(np.indices(img.shape).T - cen).T)

#Get total signal in FWHM
signal = (img[rr <= fwhm/2] - ccd_bias).sum() * ccd_gain
num_ap = img[rr <= fwhm/2].size

#Get noise
noise = np.sqrt(signal + num_ap*(ccd_dark*texp + \
    ccd_read**2. + ccd_cic))

#Compare SNR (mean per pixel)
snr = signal / noise / np.sqrt(num_ap)

print(f'\nSNR: {snr:.2f}\n')

plt.imshow(img)
breakpoint()
