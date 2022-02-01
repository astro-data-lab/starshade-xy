"""
get_lab_exposure_time.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 04-02-2021

Description: Script to calculate lab exposure time needed to reach certain SNR.
"""

import numpy as np
from imager import Imager
import matplotlib.pyplot as plt;plt.ion()

exp_time = 60

##############

#Imager parameters
params = {

    ### FLYER params ###
    'is_sim':               False,
    'do_save':              False,
    'verbose':              True,
    ### Camera params ###
    'exp_time':             exp_time,
    'num_scans':            1,
    'camera_wait_temp':     False,
    'camera_temp':          -80,
    'camera_pupil_center':  (567, 475),
    'camera_pupil_width':   250,
    'binning':              4,
    ### Motion params ###
    'zero_pos':             [2500, 7000],      #[motor steps]

}

#Load imager
imgur = Imager(params)

#Setup run
imgur.setup_run()

#Move to center
imgur.move_to_pos(0,0)

#Take picture
img = imgur.take_picture()[0]

# from astropy.io import fits
# with fits.open(r'.\new_data_30s\image__0191.fits') as hdu:
#     img = hdu[0].data[0].astype(float)

#Subtract out bias
nbk = 40//params['binning']
img -= np.median(np.concatenate((img[:nbk,:nbk],img[:nbk,-nbk:], img[-nbk:,:nbk], img[-nbk:,-nbk:])))

#Show image
plt.plot(img.shape[0]/2, img.shape[1]/2, 'r+')
plt.imshow(img)

#FWHM
fwhm = 30 / params['binning']
gain = 0.768

#Get num_pixels inside fwhm
rr = np.hypot(*(np.indices(img.shape) - len(img)/2))
num_ap = img[rr <= fwhm/2].size

#Get signal in FWHM
signal2 = img[rr <= fwhm/2] * gain

#Compute SNR proxy with mean / std dev
snr2 = signal2.mean() / signal2.std()

#Get signal in FWHM
signal = img[rr <= fwhm/2].sum() * gain

#Get noise
ccd_dark = 7e-4
ccd_read = 4.78
ccd_cic = 0.0025
noise = np.sqrt(signal + num_ap*(ccd_dark*exp_time + \
    ccd_read**2. + ccd_cic))

snr = signal/noise/np.sqrt(num_ap)


print(f'\nExposure time: {exp_time:.1f} [s], SNR: {snr:.3f} [per pixel], SNR2: {snr2:.3f}\n')

breakpoint()
