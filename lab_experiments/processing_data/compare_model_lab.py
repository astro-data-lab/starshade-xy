"""
compare_model_lab.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 04-08-2021

Description: Script to compare experimental and model pupil plane images.

"""

import numpy as np
import matplotlib.pyplot as plt;plt.ion()
from astropy.io import fits
from scipy.ndimage import shift

lab_name = './image__stable_30s.fits'
mod_name = '../diffraction_code/noisy_test/0001.npy'

exp_time = 30

#Load lab data
with fits.open(lab_name) as hdu:
    lab_imgs = hdu[0].data

# lab_img = np.median(lab_imgs, 0)
lab_img = lab_imgs[13].astype(float)

#local background
back = np.median(np.concatenate((lab_img[:50,:50], lab_img[:50,-50:], lab_img[-50:,:50], lab_img[-50:,-50:])))

lab_img -= back
lab_img /= lab_img.max()

#Load model image
mod_img = np.load(mod_name)
mod_img /= mod_img.max()

#Shift image
mod_img = shift(mod_img, (1,1), order=5)

#FWHM
fwhm = 30
gain = 0.768

#Get num_pixels inside fwhm
rr = np.hypot(*(np.indices(mod_img.shape) - len(mod_img)/2))
num_ap = mod_img[rr <= fwhm/2].size

#Get signal in FWHM
lab_signal2 = lab_img[rr <= fwhm/2] * gain
mod_signal2 = mod_img[rr <= fwhm/2] * gain

#Compute SNR proxy with mean / std dev
lab_snr2 = lab_signal2.mean() / lab_signal2.std()
mod_snr2 = mod_signal2.mean() / mod_signal2.std()

#Get signal in FWHM
lab_signal = lab_signal2.sum()
mod_signal = mod_signal2.sum()

#Get noise
ccd_dark = 7e-4
ccd_read = 3.20
ccd_cic = 0.0025
lab_noise = np.sqrt(lab_signal + num_ap*(ccd_dark*exp_time + \
    ccd_read**2. + ccd_cic))
mod_noise = np.sqrt(mod_signal + num_ap*(ccd_dark*exp_time + \
    ccd_read**2. + ccd_cic))

lab_snr = lab_signal/lab_noise/num_ap
mod_snr = mod_signal/mod_noise/num_ap

print(f'\nLAB - SNR 1: {lab_snr:.3f}, SNR 2: {lab_snr2:.3f}\n')
print(f'\nMOD - SNR 1: {mod_snr:.3f}, SNR 2: {mod_snr2:.3f}\n')

#Plots
fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(11,8))
axes[0].imshow(mod_img)
axes[1].imshow(lab_img)

lfig, laxes = plt.subplots(2, figsize=(8,11))
laxes[0].plot(mod_img[len(mod_img)//2], label='Model')
laxes[0].plot(lab_img[len(lab_img)//2], label='Lab')
laxes[1].plot(mod_img[:,len(mod_img)//2])
laxes[1].plot(lab_img[:,len(lab_img)//2])
laxes[0].legend()
breakpoint()
