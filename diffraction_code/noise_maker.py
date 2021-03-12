"""
noise_maker.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 03-11-2021

Description: Add simulated detector noise to images pre-calculated using the BDW
    diffraction code.

"""

import numpy as np
import os
import glob

class Noise_Maker(object):

    def __init__(self, params={}):
        self.set_parameters(params)

############################################
####    Initialization ####
############################################

    def set_parameters(self, params):
        #Default parameters
        def_pms = {
            ### Loading ###
            'load_dir':         '',             #Directory to load pre-made images
            'load_file_ext':    '.npy',         #File extenstion for data
            ### Saving ###
            'save_dir':         './',           #Directory to save images with noise added
            'do_save':          True,           #Save data?
            ### Observation ###
            'target_SNR':       5,              #Target SNR of peak of diffraction pattern
            'diff_peak':        3.31e-3,        #Peak of diffraction pattern from simulation calculation
            'count_rate':       100,            #Expected counts/s of peak of diffraction pattern
            ### Detector ###
            'ccd_read':         3.20,           #Detector read noise [e-/pixel/frame]
            'ccd_gain':         0.768,          #Detector inverse-gain [e-/count]
            'ccd_dark':         7e-4,           #Detector dark noise [e-/pixel/s]
            'ccd_cic':          0.0025,         #Detector CIC noise [e/pixel/frame]
            'ccd_bias':         500,            #Detector bias level [counts]
        }

        #Set user and default parameters
        for k,v in {**def_pms, **params}.items():
            #Check if valid parameter name
            if k not in def_pms.keys():
                print(f'\nError: Invalid Parameter Name: {k}\n')
                import sys; sys.exit(0)

            #Set parameter value as attribute
            setattr(self, k, v)

        #Create save directory
        if self.do_save:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

############################################
############################################

############################################
####    Main Script ####
############################################

    def run_script(self):
        #Get peak count estimate from SNR
        peak_cnts = self.get_counts_from_SNR()

        #Get all filenames
        self.image_files = glob.glob(f'{self.load_dir}/*{self.load_file_ext}')

        #Loop through and process each image file
        for img_file in self.image_files:

            #Load image
            img = np.load(img_file)

            #Add noise to image
            img = self.add_noise_to_image(img, peak_cnts)

            #Save image
            if self.do_save:
                save_name = img_file.split('/')[-1].split('.npy')[0]
                np.save(f'{self.save_dir}/{save_name}', img)

############################################
############################################

############################################
####   SNR ####
############################################

    def get_counts_from_SNR(self):
        #Build SNR for full range of counts
        sig_count = np.arange(2**16)
        texp = sig_count / self.count_rate
        noise = np.sqrt(sig_count*self.ccd_gain + self.ccd_dark*texp + \
            self.ccd_read**2. + self.ccd_cic)
        snr = sig_count * self.ccd_gain / noise

        #Find number of counts that corresponds to target SNR
        counts = sig_count[np.argmin(np.abs(snr - self.target_SNR))]

        return counts

    def check_snr(self, img, texp):
        #Get max signal
        signal = np.mean((img.max() - self.ccd_bias) * self.ccd_gain)

        #Get noise
        noise = np.sqrt(signal + self.ccd_dark*texp + \
            self.ccd_read**2. + self.ccd_cic)

        #Compare SNR
        snr = signal / noise

        print(f'SNR: {snr:.2f}, Target SNR: {self.target_SNR:.2f}')

        import matplotlib.pyplot as plt; plt.ion()
        plt.cla()
        plt.imshow(img)

        breakpoint()

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
        img = np.random.poisson(img).astype(float)

        #Dark noise
        exp_time = peak_cnts / self.count_rate
        img += np.random.poisson(self.ccd_dark*exp_time, size=img.shape)

        #CIC noise
        img += np.random.poisson(self.ccd_cic, size=img.shape)

        #Readout noise on top of bias
        img += np.random.normal(self.ccd_bias*self.ccd_gain, self.ccd_read, size=img.shape)

        #Add conventional gain (= CCD sensitivity [e-/count])
        img /= self.ccd_gain

        #Round to convert to counts
        img = np.round(img)

        #Check SNR
        # self.check_snr(img, exp_time)

        #Remove bias
        img -= self.ccd_bias

        return img

############################################
############################################
