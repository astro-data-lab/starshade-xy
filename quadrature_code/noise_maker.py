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
            'load_dir_base':    './',           #Directory to load pre-made images
            'load_file_ext':    '.npy',         #File extenstion for data
            'base_name':        '',             #Base image name
            ### Saving ###
            'save_dir_base':    './',           #Directory to save images with noise added
            'do_save':          True,           #Save data?
            ### Observation ###
            'target_SNR':       5,              #Target SNR of peak of diffraction pattern
            'multi_SNRs':       [],             #List of target SNR
            'diff_peak':        3.31e-3,        #Peak of diffraction pattern from simulation calculation
            'count_rate':       100,            #Expected counts/s of peak of diffraction pattern
            'peak2mean':        0.67,           #Conversion from peak counts to mean counts in FWHM for J_0^2
            'fwhm':             10,             #Full-width at half-maximum of J_0^2
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

        #Build directories
        self.load_dir = f'{self.load_dir_base}/{self.base_name}'
        self.save_dir = f'{self.save_dir_base}/{self.base_name}'

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
        peak_cnts, exp_time = self.get_counts_from_SNR(self.target_SNR)

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

    def run_multiple_snr_script(self):
        #Get all filenames
        self.image_files = glob.glob(f'{self.load_dir}/*{self.load_file_ext}')

        #Save info
        if self.do_save:
            with h5py.File(f'{self.save_dir}/meta.h5', 'w') as f:
                f.create_dataset('multi_SNRs', data=self.multi_SNRs)

        #Image counter
        cntr = 0

        #Get all shifts
        shifts = np.genfromtxt(f'{self.load_dir}/{self.base_name}.csv', delimiter=',')

        #Open csv file
        if self.do_save:
            csv_file = open(f'{self.save_dir}/{self.base_name}.csv', 'w')
            atexit.register(csv_file.close)

        #Loop through SNRs
        for snr in self.multi_SNRs:
            print(f'Adding noise to SNR: {snr}')

            #Get peak count estimate from SNR
            peak_cnts, exp_time = self.get_counts_from_SNR(snr)

            #Loop through and process each image file
            for i in range(len(self.image_files))[:1]:

                #Load image
                img = np.load(self.image_files[i])

                #Add noise to image
                img = self.add_noise_to_image(img, peak_cnts)
                import matplotlib.pyplot as plt;plt.ion()
                plt.imshow(img)
                # print(img.max(), peak_cnts, exp_time, peak_cnts/exp_time)
                print(img.max() / exp_time)
                breakpoint()

                #Normalize by exposure time
                img /= exp_time

                #Save
                if self.do_save:
                    #Save image
                    np.save(f'{self.save_dir}/{str(cntr).zfill(6)}', img)

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
        num_ap = (self.fwhm - 1)**2

        #Total counts in signal
        total_sig = np.arange(2**16*num_ap).astype(float)

        #Exposure time to get total counts (count rate is peak count rate)
        texps = total_sig / (num_ap * self.peak2mean * self.count_rate)

        #Noise for each exposure time
        noise = np.sqrt(total_sig*self.ccd_gain + num_ap*(self.ccd_dark*texps + \
            self.ccd_read**2. + self.ccd_cic))

        #Mean SNR
        mean_snr = total_sig * self.ccd_gain / noise / num_ap

        #Get exposure time from best fit to mean SNR
        exp_time = texps[np.argmin(np.abs(mean_snr - target_SNR))]

        #Get peak counts to aim for
        peak_counts = self.count_rate * exp_time

        return peak_counts, exp_time

    def check_snr(self, img, texp):
        #Get radius of image around peak
        cen = np.array(np.unravel_index(np.argmax(img), img.shape))
        rr = np.hypot(*(np.indices(img.shape).T - cen).T)

        #Get total signal in FWHM
        signal = (img[rr <= self.fwhm/2] - self.ccd_bias).sum() * self.ccd_gain
        num_ap = img[rr <= self.fwhm/2].size

        #Get noise
        noise = np.sqrt(signal + num_ap*(self.ccd_dark*texp + \
            self.ccd_read**2. + self.ccd_cic))

        #Compare SNR (mean per pixel)
        snr = signal / noise / num_ap

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

        #Check SNR (uncomment)
        # self.check_snr(img, exp_time)

        #Remove bias
        img -= self.ccd_bias

        return img

############################################
############################################
