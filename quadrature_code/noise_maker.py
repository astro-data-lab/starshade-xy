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
import h5py
import atexit
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
            ### Loading ###
            'load_dir_base':    '',             #Directory to load pre-made images
            'load_file_ext':    '.npy',         #File extenstion for data
            'base_name':        '',             #Base image name
            ### Saving ###
            'save_dir_base':    '',             #Directory to save images with noise added
            'do_save':          True,           #Save data?
            'is_test':          False,          #Is this test of SNR?
            ### Observation ###
            'target_SNR':       5,              #Target SNR of peak of diffraction pattern
            'multi_SNRs':       [],             #List of target SNR
            'diff_peak':        3.615e-3,       #Peak of diffraction pattern from simulation calculation
            'count_rate':       7,            #Expected counts/s of peak of diffraction pattern
            'peak2mean':        0.68,           #Conversion from peak counts to mean counts in FWHM for J_0^2
            'fwhm':             28,             #Full-width at half-maximum of J_0^2
            ### Detector ###
            'ccd_read':         4.78,           #Detector read noise [e-/pixel/frame]
            'ccd_gain':         0.768,          #Detector inverse-gain [e-/count]
            'ccd_dark':         7e-4,           #Detector dark noise [e-/pixel/s]
            'ccd_cic':          0.0025,         #Detector CIC noise [e/pixel/frame]
            'ccd_bias':         500,            #Detector bias level [counts]
            ### Numerics ###
            'seed':             None,           #Seed for random number generator
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
        self.rng = np.random.default_rng(self.seed)

        #Don't save with test
        if self.is_test:
            self.do_save = False

        #Build directories
        self.load_dir = os.path.join('Simulated_Images', self.load_dir_base, self.base_name)
        self.save_dir = os.path.join('Simulated_Images', self.save_dir_base, self.base_name)

        #Create save directory
        if self.do_save:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        #Get calibration value
        if not self.is_test:
            cal_img = np.load(os.path.join(self.load_dir, 'calibration.npy'))
            self.suppression_norm = cal_img[cal_img > 0].mean()

############################################
############################################

############################################
####    Main Script ####
############################################

    def run_script(self):
        #Get peak count estimate from SNR
        peak_cnts, exp_time = self.get_counts_from_SNR(self.target_SNR)

        #Get all filenames
        self.image_files = glob.glob(os.path.join(self.load_dir, \
            '*' + self.load_file_ext))

        #Loop through and process each image file
        for img_file in self.image_files:

            #Load image
            img = np.load(img_file)

            #Add noise to image
            img = self.add_noise_to_image(img, peak_cnts)

            #Save image
            if self.do_save:
                save_name = os.path.split(img_file)[-1].split('.npy')[0]
                np.save(os.path.join(self.save_dir, save_name), img)

    ############################################

    def run_multiple_snr_script(self):

        #Image counter
        cntr = 0

        #Get all shifts
        shifts = np.genfromtxt(os.path.join(self.load_dir, \
            self.base_name + '.csv'), delimiter=',')

        #Number of images
        num_imgs = len(shifts)

        #Get exposure times, peak_cnts
        peak_exps = np.array([self.get_counts_from_SNR(snr) for snr in self.multi_SNRs])

        #Save info
        if self.do_save:
            with h5py.File(os.path.join(self.save_dir, 'meta.h5'), 'w') as f:
                f.create_dataset('multi_SNRs', data=self.multi_SNRs)
                f.create_dataset('peak_cnts', data=peak_exps[:,0])
                f.create_dataset('exp_times', data=peak_exps[:,1])
                f.create_dataset('num_imgs', data=num_imgs)

        #Open csv file
        if self.do_save:
            csv_file = open(os.path.join(self.save_dir, self.base_name+'.csv'), 'w')
            atexit.register(csv_file.close)

        #Loop through SNRs
        for snr_i in range(len(self.multi_SNRs)):

            snr = self.multi_SNRs[snr_i]
            print(f'Adding noise to SNR: {snr}')

            #Just for debugging
            self.target_SNR = snr

            #Get peak count estimate from SNR
            peak_cnts, exp_time = peak_exps[snr_i]

            #Loop through and process each image file
            for i in range(num_imgs):

                #Load image
                img = np.load(os.path.join(self.load_dir, str(i).zfill(6)+'.npy'))

                #Add noise to image
                img = self.add_noise_to_image(img, peak_cnts)

                #Convert to suppression
                img *= self.diff_peak / (self.count_rate * exp_time * self.suppression_norm)

                #Save
                if self.do_save:
                    #Save image
                    np.save(os.path.join(self.save_dir, str(cntr).zfill(6)), img)

                    #Writeout shift
                    csv_file.write(f'{str(cntr).zfill(6)}, {shifts[i][1]}, {shifts[i][2]}\n')

                #Increment
                cntr += 1

############################################
############################################

############################################
####   SNR ####
############################################

    def get_counts_from_SNR(self, target_SNR):

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
        func = lambda s: np.abs(mean_snr(s) - target_SNR)
        x0 = num_ap*self.fwhm       #Start large
        out = fmin(func, x0, disp=0)[0]

        #Get exposure time
        exp_time = texp(out)

        #Get peak counts to aim for
        peak_counts = self.count_rate * exp_time

        return peak_counts, exp_time

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

############################################
####    SNR Test ####
############################################

    def run_snr_test(self, in_img):

        #Get exposure times, peak_cnts
        peak_exps = np.array([self.get_counts_from_SNR(snr) for snr in self.multi_SNRs])

        #Loop through SNRs
        for snr_i in range(len(self.multi_SNRs)):

            snr = self.multi_SNRs[snr_i]
            print(f'Adding noise to SNR: {snr}')

            #Just for debugging
            self.target_SNR = snr

            #Get peak count estimate from SNR
            peak_cnts, exp_time = peak_exps[snr_i]

            #Add noise to image
            img = self.add_noise_to_image(in_img.copy(), peak_cnts)

            #Get SNR
            calc_snr = self.check_snr(img, exp_time)

            #Assert is close (15%)
            assert(abs(calc_snr - snr)/snr < 0.15)

            #Plot
            if [False, True][0]:
                print(f'SNR: {calc_snr:.2f}, Target SNR: {self.target_SNR:.2f}, Exp Time: {exp_time:.2f} [s]')

                import matplotlib.pyplot as plt; plt.ion()
                plt.cla()
                plt.imshow(img)
                breakpoint()

    def check_snr(self, img, texp):
        #Get radius of image around peak
        cen = np.array(np.unravel_index(np.argmax(img), img.shape))
        rr = np.hypot(*(np.indices(img.shape).T - cen).T)

        #Get total signal in FWHM
        signal = img[rr <= self.fwhm/2].sum() * self.ccd_gain
        num_ap = np.pi*(self.fwhm/2)**2

        #Get noise
        noise = np.sqrt(signal + num_ap*(self.ccd_dark*texp + \
            self.ccd_read**2. + self.ccd_cic))

        #Compare SNR (mean per pixel)
        snr = signal / noise / num_ap

        return snr

############################################
############################################
