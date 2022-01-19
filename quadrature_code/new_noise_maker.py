"""
new_noise_maker.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 01-19-2022

Description: Add simulated detector noise to images pre-calculated using the BDW
    diffraction code.

"""

import numpy as np
import os
import h5py
from scipy.optimize import fmin

class Noise_Maker(object):

    def __init__(self, params={}):
        self.set_parameters(params)

############################################
####    Initialization ####
############################################

    def set_parameters(self, params):
        #Default parameters
        def_pms = {
            ### Testing ###
            'is_test':          False,          #Is this test of SNR?
            ### Observation ###
            'diff_peak':        3.615e-3,       #Peak of diffraction pattern from simulation calculation
            'count_rate':       7,              #Expected counts/s of peak of diffraction pattern
            'peak2mean':        0.68,           #Conversion from peak counts to mean counts in FWHM for J_0^2
            'fwhm':             28,             #Full-width at half-maximum of J_0^2
            'snr_range':        [0.01,10],
            'n_snrs':           500,
            ### Detector ###
            'ccd_read':         4.78,           #Detector read noise [e-/pixel/frame]
            'ccd_gain':         0.768,          #Detector inverse-gain [e-/count]
            'ccd_dark':         7e-4,           #Detector dark noise [e-/pixel/s]
            'ccd_cic':          0.0025,         #Detector CIC noise [e/pixel/frame]
            'ccd_bias':         500,            #Detector bias level [counts]
            ### Numerics ###
            'seed':             None,           #Seed for random number generator
            'rng':              None,           #Already started Random Number Generator
        }

        #Set user and default parameters
        for k,v in {**def_pms, **params}.items():
            #Check if valid parameter name
            if k not in def_pms.keys():
                print(f'\nError: Invalid Parameter Name: {k}\n')
                import sys; sys.exit(0)

            #Set parameter value as attribute
            setattr(self, k, v)

        #Random number generator
        if self.rng is None:
            self.rng = np.random.default_rng(self.seed)

        #Create peak_cnts, exp_time vs SNR curve
        self.create_snr_curve()

############################################
############################################

############################################
####    Main Script ####
############################################

    def add_noise(self, img, snr):

        #Get peak counts and exposure time
        peak_cnts = np.interp(snr, self.snrs, self.peak_counts)
        exp_time = np.interp(snr, self.snrs, self.exp_times)

        #Add noise to image
        img = self.add_noise_to_image(img, peak_cnts)

        #Convert to suppression
        img *= self.diff_peak / (self.count_rate * exp_time * self.suppression_norm)

        return img

    def set_suppression_norm(self, cal_img):
        self.suppression_norm = cal_img[cal_img > 0].mean()

############################################
############################################

############################################
####   SNR ####
############################################

    def create_snr_curve(self):
        #Range of snrs
        snrs = np.linspace(self.snr_range[0], self.snr_range[1], self.n_snrs)

        #Estimate number of points in FWHM
        num_ap = np.pi*(self.fwhm/2)**2

        #Exposure time to get total counts (count rate is peak count rate)
        texp = lambda s: s / (num_ap * self.peak2mean * self.count_rate)

        #Noise for each exposure time
        noise = lambda s: np.sqrt(s*self.ccd_gain + num_ap*(self.ccd_dark*texp(s) + \
            self.ccd_read**2. + self.ccd_cic))

        #Mean SNR
        mean_snr = lambda s: s * self.ccd_gain / noise(s) / num_ap

        #Solve for total counts that gives mean SNR closest to target SNR
        func = lambda s, target_SNR: np.abs(mean_snr(s) - target_SNR)
        x0 = num_ap*self.fwhm       #Start large

        outs = np.array([])
        for s in snrs:
            outs = np.concatenate((outs, [fmin(func, x0, args=(s,), disp=0)[0]]))
            x0 = outs[-1]

        #Get exposure time
        exp_times = texp(outs)

        #Get peak counts to aim for
        peak_counts = self.count_rate * exp_times

        #Store
        self.snrs = snrs
        self.exp_times = exp_times
        self.peak_counts = peak_counts

############################################
############################################

############################################
####    Noise Model ####
############################################

    def add_noise_to_image(self, img, peak_cnts):

        #Turn into counts that would give target SNR
        img *= peak_cnts / self.diff_peak

        #Convert to photons
        img = np.round(img * self.ccd_gain)

        #Photon noise
        img = self.rng.poisson(img).astype(float)

        #Dark noise
        exp_time = peak_cnts / self.count_rate
        img += self.rng.poisson(self.ccd_dark*exp_time, size=img.shape)

        #CIC noise
        img += self.rng.poisson(self.ccd_cic, size=img.shape)

        #Readout noise on top of bias
        img += self.rng.normal(self.ccd_bias*self.ccd_gain, self.ccd_read, size=img.shape)

        #Add conventional gain (= CCD sensitivity [e-/count])
        img /= self.ccd_gain

        #Round to convert to counts
        img = np.round(img)

        ##Check SNR (uncomment)
        # self.check_snr(img, exp_time)

        #Remove bias
        img -= self.ccd_bias

        return img

############################################
############################################
