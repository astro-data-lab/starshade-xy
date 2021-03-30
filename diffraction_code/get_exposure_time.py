"""
get_exposure_time.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 03-22-2021

Description: Script to estimate exposure time needed to reach target SNR.

"""

from bdw import BDW
from noise_maker import Noise_Maker
import numpy as np
import matplotlib.pyplot as plt;plt.ion()

#Specify simulation parameters
params = {
    ### Lab ###
    'wave':             0.405e-6,       #Wavelength of light [m]
    'z0':               27.5,           #Source - starshade distance [m]
    'z1':               50.,            #Starshade - telescope distance [m]

    ### Telescope ###
    'tel_diameter':     2.4e-3,         #Telescope aperture diameter [m]
    'num_tel_pts':      32,             #Size of grid to calculate over pupil
    'image_pad':        0,
    'tel_shift':        [0, 0],         #(x,y) shift of telescope relative to starshade-source line [m]

    ### Starshade ###
    'apod_name':        'lab_ss',       #Apodization profile name. Options: ['lab_ss', 'circle']
    'with_spiders':     False,          #Superimpose secondary mirror spiders on pupil image?

    ### Saving ###
    'do_save':          False,          #Save data?
}

target_SNR = 5


#Load BDW class
bdw = BDW(params)

#Run simulation
img = bdw.run_sim()

#Turn into intensity
img = np.abs(img)**2

#Load instance of noise maker
maker = Noise_Maker()

plt.imshow(img)

#FWHM
fwhm = 2*(np.abs(np.argmin(np.abs(img[len(img)//2] - img.max()/2)) - len(img)/2))

#Get num_pixels inside fwhm
rr = np.hypot(*(np.indices(img.shape) - len(img)/2))
num_ap = img[rr <= fwhm/2].size

#Get conversion from peak to mean in FWHM
peak2mean = img[rr <= fwhm/2].mean() / img.max()

#Total counts in signal
total_sig = np.arange(2**16*num_ap).astype(float)

#Expsoure time to get total counts (count rate is peak count rate)
texps = total_sig / (num_ap * peak2mean * maker.count_rate)

#Noise for each exposure time
noise = np.sqrt(total_sig*maker.ccd_gain + num_ap*(maker.ccd_dark*texps + \
    maker.ccd_read**2. + maker.ccd_cic))

#Mean SNR
mean_snr = total_sig * maker.ccd_gain / noise / num_ap

#Get exposure time from best fit to mean SNR
exp_time = texps[np.argmin(np.abs(mean_snr - target_SNR))]

#Get peak counts to aim for
peak_counts = maker.count_rate * exp_time

#Get noise image
nsy_img = maker.add_noise_to_image(img.copy(), peak_counts)

def check_snr_peak(img, texp):
    #Get max signal
    signal = img.max() * maker.ccd_gain

    #Get noise
    noise = np.sqrt(signal + maker.ccd_dark*texp + \
        maker.ccd_read**2. + maker.ccd_cic)

    #Compare SNR
    snr = signal / noise

    print(f'Peak SNR: {snr:.2f}')

def check_snr_mean(img, texp):
    #Get total signal
    signal = img[rr <= fwhm/2].sum() * maker.ccd_gain
    nap = img[rr <= fwhm/2].size

    #Get noise
    noise = np.sqrt(signal + nap*(maker.ccd_dark*texp + \
        maker.ccd_read**2. + maker.ccd_cic))

    #total SNR
    snr = signal / noise

    #Mean
    mean_snr = snr / nap

    print(f'Total SNR: {snr:.2f}')
    print(f'Mean SNR: {mean_snr:.2f}')

print(f'Target SNR: {target_SNR:.0f}')
check_snr_peak(nsy_img, exp_time)
check_snr_mean(nsy_img, exp_time)

plt.figure()
plt.imshow(nsy_img)

breakpoint()
