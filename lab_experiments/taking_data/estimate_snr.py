"""
estimate_snr.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 04-05-2021

Description: Script to estimate the SNR of experimental images.

"""

import numpy as np
import matplotlib.pyplot as plt;plt.ion()
from astropy.io import fits
import h5py
import glob
import image_util
from scipy.ndimage import affine_transform

all_sessions = ['/run__6_01_21/data_1s_bin1']

is_median = False

for session in all_sessions:

    #Choose aperture size for proper scaling to space
    num_pts = 96
    image_pad = 10

    #aperture size (from pupil magnification)
    Dtel = num_pts * 1.764*13e-6

    #TODO: hide
    data_dir = f'/home/aharness/Research/Frick_Lab/Data/FFNN/{session}'

    def get_image(inum):
        with fits.open(f'{data_dir}/image__{str(inum).zfill(4)}.fits') as hdu:
            data = hdu[0].data.astype(float)
            exp = hdu[0].header['EXPOSURE']

        return data, exp

    #Read record
    record = np.genfromtxt(f'{data_dir}/record.csv', delimiter=',')

    #Get centered image
    ind0 = int(record[:,0][np.argmin(np.hypot(record[:,1], record[:,2]))])
    img, exp_time = get_image(ind0)
    img_shp = img.shape[-2:]
    if is_median:
        img = np.median(img,0)
    else:
        img = img[0]

    #Get binning
    nbin = int(np.round(250/img_shp[0]))
    num_pts //= nbin

    #FWHM
    fwhm = 30 / nbin
    gain = 0.768

    #Get num_pixels inside fwhm
    rr = np.hypot(*(np.indices(img.shape) - len(img)/2))
    finds = rr <= fwhm/2
    num_ap = img[finds].size

    #Get signal in FWHM
    signal2 = img[finds] * gain

    #Compute SNR proxy with mean / std dev
    snr2 = signal2.mean() / signal2.std()

    #Get signal in FWHM
    signal = img[finds].sum() * gain

    #Get noise
    ccd_dark = 7e-4
    ccd_read = 4.78
    ccd_cic = 0.0025
    noise = np.sqrt(signal + num_ap*(ccd_dark*exp_time + \
        ccd_read**2. + ccd_cic))

    snr = signal/noise/np.sqrt(num_ap)

    print(f'\nExposure time: {exp_time:.1f} [s], SNR: {snr:.3f}, SNR 2: {snr2:.3f}\n')

    img[finds] = 0
    plt.cla()
    plt.imshow(img)

    breakpoint()
