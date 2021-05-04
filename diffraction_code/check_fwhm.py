"""
check_fwhm.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 03-22-2021

Description: Script to estimate size of FHWM on pupil.

"""

from bdw import BDW
import numpy as np
import h5py
import matplotlib.pyplot as plt;plt.ion()
from scipy.special import j0

is_wfirst = [False,True][0]

if is_wfirst:
    z0 = 1e19
    z1 = 26e6
    telD = 2.4
    srad = 13
    apod = 'wfirst'

else:
    z0 = 27.5
    z1 = 50.
    telD = 2.2e-3
    srad = 12.53e-3
    apod = 'lab_ss'

#Specify simulation parameters
params = {
    ### Lab ###
    'wave':             0.405e-6,      #Wavelength of light [m]
    'z0':               z0,            #Source - starshade distance [m]
    'z1':               z1,            #Starshade - telescope distance [m]

    ### Telescope ###
    'tel_diameter':     telD,            #Telescope aperture diameter [m]
    'num_tel_pts':      220,             #Size of grid to calculate over pupil
    'image_pad':        0,
    'tel_shift':        [0, 0],         #(x,y) shift of telescope relative to starshade-source line [m]

    ### Starshade ###
    'apod_name':        apod,           #Apodization profile name. Options: ['lab_ss', 'circle']
    'with_spiders':     False,          #Superimpose secondary mirror spiders on pupil image?

    ### Saving ###
    'do_save':          True,          #Save data?
    'save_dir_base':    './data',
    'save_ext':         'fwhm' + ['','_wfirst'][int(is_wfirst)],
}

#Load BDW class
bdw = BDW(params)

if [False, True][0]:

    #Run simulation
    img = bdw.run_sim()

else:

    #Load data
    save_dir = params['save_dir_base']
    save_ext = params['save_ext']
    with h5py.File(f'{save_dir}/pupil_{save_ext}.h5', 'r') as f:
        img = np.abs(f['pupil_Ec'][()])**2
        xx = f['pupil_xx'][()]

    img /= img.max()

    #find fhwm
    fwhm = 2*abs(xx[np.argmin(np.abs(img[len(img)//2] - 0.5))])

    #get numerical constant in fwhm
    ss = np.linspace(0, 2, 1000)
    gamma = ss[np.argmin(np.abs(j0(ss)**2 - 0.5))]

    #True fwhm (1.13 comes from fwhm of j0^2)
    tru_fwhm = bdw.wave*bdw.zeff/(np.pi*srad)*gamma

    #Get effective radius
    reff = bdw.wave*bdw.zeff/(np.pi*fwhm)*gamma

    tf2 = bdw.wave*bdw.zeff/(np.pi*reff)*gamma

    print(f'\nFWHM: {fwhm:.1e}, True: {tru_fwhm:.1e}, Reff: {reff:.3e}\n')

    plt.figure()
    plt.plot(xx, img[len(img)//2])
    plt.axvline(fwhm/2, color='k')
    plt.axvline(tru_fwhm/2, color='c')
    plt.axvline(tf2/2, color='r', linestyle=':')
    plt.axhline(0.5, color='k')
    breakpoint()
