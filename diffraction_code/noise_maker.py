"""
noise_maker.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 03-11-2021

Description: Add simulated detector noise to images pre-calculated using the BDW
    diffraction code.

"""

import numpy as np
import matplotlib.pyplot as plt
import os
import glob

plt.ion()

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
            'load_file_ext':    '',             #File extenstion for data
            ### Saving ###
            'save_dir':         './',           #Directory to save images with noise added
            'do_save':          True,           #Save data?
            ### Observation ###
            'peak_SNR':         5,              #SNR of peak of diffraction pattern
            'diff_peak':        0.84,           #Peak of diffraction pattern from simulation calculation
            ### Detector ###
            'det_QE':           0.55,           #Detector quantum efficiency
            'det_read':         3.20,           #Detector read noise [e-/pixel/frame]
            'det_gain':         0.768,          #Detector inverse-gain [e-/count]
            'det_dark':         7e-4,           #Detector dark noise [e-/pixel/s]
            'det_cic':          0.0025,         #Detector CIC noise [e/pixel/frame]
            'det_bias':         500,            #Detector bias level [counts]
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
        # peak_cnts = self.get_counts_from_SNR()

        #Get all filenames
        self.image_files = glob.glob(f'{self.load_dir}/*{self.load_file_ext}')

        #Loop through and process each image file
        for img_file in self.image_files:

            #Load image
            img = plt.imread(img_file, format='png')

            #Convert to grayscale
            # img = img[...,:3].dot([0.2989, 0.5870, 0.1140])

            #Add noise to image
            # img = self.add_noise_to_image(img, peak_cnts)

            plt.imshow(img)
            print(img.max((0,1)))

            breakpoint()

############################################
############################################

############################################
####    Noise Model ####
############################################

    def get_counts_from_SNR(self):

        breakpoint()

    def add_noise_to_image(self, img, peak_cnts):

        breakpoint()
        #Convert to photons
        img = np.round(img.T * self.flux_to_photons * exp_time)

        #Photon noise
        if with_noise:
            img = np.random.poisson(img).astype(float)

        #Dark noise
        if with_noise:
            img += np.random.poisson(self.gpilot.dark_noise*exp_time,size=img.shape)

        #CIC noise
        if with_noise:
            img += np.random.poisson(self.gpilot.cic_noise,size=img.shape)

        #Add uncertainty in EM gain generation with Gamma distribution (Maxime), else add conv gain
        if with_noise:
            if self.gpilot.camera_is_EM:
                img[img > 0.] = np.random.gamma(img[img > 0.],self.gpilot.camera_EM_gain)

        #Readout noise on top of bias
        if with_noise:
            img += np.random.normal(self.gpilot.ccd_bias*self.gpilot.hardware.camera_gain, \
                self.gpilot.hardware.read_noise,size=img.shape)
        else:
            img += self.gpilot.ccd_bias * self.gpilot.hardware.camera_gain

        #Add conventional gain (= CCD sensitivity [e-/count])
        img /= self.gpilot.hardware.camera_gain

        #Remove EM gain
        if self.gpilot.camera_is_EM:
            img += self.gpilot.ccd_bias*(self.gpilot.camera_EM_gain - 1.)
            img /= self.gpilot.camera_EM_gain

        #Round to convert to counts
        img = np.round(img)

        plt.imshow(img)
        breakpoint()

############################################
############################################
