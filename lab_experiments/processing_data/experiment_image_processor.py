"""
experiment_image_processor.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 06-28-2021

Description: Script to add spiders and secondary to experimental images.

"""

import numpy as np
import os
import h5py
import image_util
import time
import photo_functions as pfunc
from scipy.ndimage import affine_transform
from truth_sensor import Truth_Sensor

class Experiment_Image_Processor(object):

    def __init__(self, params={}):
        self.set_parameters(params)
        self.setup()

############################################
####    Initialization ####
############################################

    def set_parameters(self, params):
        #Default parameters
        def_pms = {
            ### Loading ###
            'base_dir':         '/home/aharness/Research/Frick_Lab/Data/FFNN',
            'xtras_dir':        '../../quadrature_code/xtras',
            'save_dir':         'Processed_Images',
            'run':              '',
            'session':          '',
            'do_round_mask':    False,
            ### Saving ###
            'do_save':          False,
            'do_plot':          False,
            ### Observation ###
            'is_med':           False,
            'base_num_pts':     96,
            'image_pad':        10,
            ### Truth Sensor ###
            'image_center':     None,
            'sensing_method':   'model',        #Options: ['model', 'centroid']
            'cen_threshold':    0.75,           #Centroiding threshold
            'wave':             403e-9,
            'ss_radius':        10.1e-3*np.sqrt(680/638),
            'z1':               50.,
            'ccd_dark':         7e-4,           #Dark noise [e/px/s]
            'ccd_read':         3.20,           #Read noise [e/px/frame]
            'ccd_cic':          0.0025,         #CIC noise [e/px/frame]
            'ccd_gain':         0.768,          #inverse gain [ct/e]
        }

        #Set user and default parameters
        for k,v in {**def_pms, **params}.items():
            #Check if valid parameter name
            if k not in def_pms.keys():
                print(f'\nError: Invalid Parameter Name: {k}\n')
                import sys; sys.exit(0)

            #Set parameter value as attribute
            setattr(self, k, v)

        #Directories
        self.load_dir = os.path.join(self.base_dir, self.session, self.run)

        #FIXED
        pupil_mag = 1.764
        pixel_size = 13e-6
        self.num_pixel_base = 250

        #Derived
        self.tel_diameter = self.base_num_pts * pupil_mag*pixel_size

    def setup(self):

        #Get image shape
        img, _, _ = pfunc.load_image(os.path.join(self.load_dir, 'image__0001.fits'))
        self.num_kin = img.shape[0]
        self.img_shape = img.shape[1:]

        #Get binning
        self.binning = int(np.round(self.num_pixel_base/self.img_shape[0]))
        self.num_pts = self.base_num_pts // self.binning

        #Load truth sensor
        self.truth = Truth_Sensor(self)

############################################
############################################

############################################
####    Main Script ####
############################################

    def run_script(self):

        #Load calibration
        self.load_calibration()

        #Load image record
        self.record = np.genfromtxt(os.path.join(self.load_dir, 'record.csv'), delimiter=',')

        #Prepare true position
        self.true_position = np.empty((len(self.record),2))

        #Run None mask
        self.process_images('none')

        #Run spiders mask
        self.process_images('spiders')

        #Run round mask
        if self.do_round_mask:
            self.process_images('round')

    ############################################

    def process_images(self, mask_type):

        #Startup
        tik = time.perf_counter()
        print(f'Running Mask: {mask_type}')

        #Load mask
        spiders = self.load_pupil_mask(mask_type)
        nspid = np.count_nonzero(spiders)

        #Loop through steps and get images and exposure times + backgrounds
        imgs = np.empty((0,) + self.img_shape)
        phos = np.empty((0,))
        locs = np.empty((0, 2))
        meta = np.empty((0, 3))
        for i in range(len(self.record)):

            #Get image
            img, exp, pho = self.get_image(int(self.record[i][0]))

            #Get excess regions
            nbk = 40//self.binning
            excess = np.concatenate((img[:,:nbk,:nbk], img[:,:nbk,-nbk:], \
                img[:,-nbk:,:nbk], img[:,-nbk:,-nbk:])).flatten()

            #Add noise in spiders and outside aperture
            img[:,spiders] = np.random.choice(excess, (img.shape[0], nspid))

            #Subtract background
            back = np.median(excess)
            img -= back

            #Take median
            if self.is_med:
                nframe = img.shape[0]
                img = np.array([np.median(img, 0)])
                pho = np.array([np.median(pho)])
            else:
                nframe = 1

            #Get current position
            if mask_type == 'none':
                #Position guess (flip x-sign b/c optics)
                pos0 = self.record[i][1:] * np.array([-1, 1])
                #Get true position
                pos = self.truth.get_position(img, exp, pos0)
                #Save
                self.true_position[i] = pos
            else:
                #Get stored true position
                pos = self.true_position[i]

            #Plot
            if self.do_plot:
                import matplotlib.pyplot as plt;plt.ion()
                print(i, pos*1e3)
                for j in range(img.shape[0]):
                    plt.cla()
                    cimg = image_util.crop_image(img[j], None, self.num_pts//2+self.image_pad)
                    plt.imshow(cimg)
                    print(cimg.max())
                    breakpoint()

            #Store images + positions
            imgs = np.concatenate((imgs, img))
            phos = np.concatenate((phos, pho))
            locs = np.concatenate((locs, [pos]*img.shape[0]))
            #Store exposure time + backgrounds + number of frames
            meta = np.concatenate((meta, [[exp, back, nframe]]*img.shape[0]))

        #Trim images
        if mask_type != 'none':
            imgs = image_util.crop_image(imgs, None, self.num_pts//2+self.image_pad)

        #Save Data
        if self.do_save:
            self.save_data(mask_type, imgs, phos, locs, meta)
            if mask_type == 'none':
                self.save_truths()

        #End
        tok = time.perf_counter()
        print(f'Time: {tok-tik:.1f} [s]\n')

############################################
############################################

############################################
####    Misc Functions ####
############################################

    def load_calibration(self):

        #Get photometer data
        cal_dir = self.load_dir.split(self.run)[0]
        self.photo_data = pfunc.load_photometer_data(cal_dir, None)

        #Load calibration data
        fname = os.path.join(cal_dir, 'cal_sup.fits')
        cimg, cexp, cpho = pfunc.get_image_data(fname, self.photo_data)

        #Kluge for 6_1_21 b/c overexposed data
        if self.session == 'run__6_01_21':
            cimg /= 0.88

        #Subtract background
        cimg -= np.median(cimg)

        #Get normalized median
        med = np.median(cimg / cpho[:,None,None], 0)

        #Threshold image
        med[med < med.mean()] = 0

        #Get calibration value from mean suppression (should equal diverging beam factor**2)
        self.cal_value = med[med != 0].mean()

    def get_image(self, inum):
        #Get data
        fname = os.path.join(self.load_dir, f'image__{str(inum).zfill(4)}.fits')
        img, exp, pho = pfunc.get_image_data(fname, self.photo_data)

        #Normalize by photometer data and suppression mean (diverging beam factor**2)
        img /= pho[:,None,None] * self.cal_value

        return img, exp, pho

    def save_data(self, mask_type, imgs, phos, locs, meta):

        ext = ['', '__median'][int(self.is_med)]
        fname = os.path.join(self.save_dir, \
            f'{self.session}__{self.run}__{mask_type}{ext}.h5')

        with h5py.File(fname, 'w') as f:
            f.create_dataset('cal_value', data=self.cal_value)
            f.create_dataset('num_tel_pts', data=self.num_pts)
            f.create_dataset('base_num_pts', data=self.base_num_pts)
            f.create_dataset('binning', data=self.binning)
            f.create_dataset('image_pad', data=self.image_pad)
            f.create_dataset('tel_diameter', data=self.tel_diameter)
            f.create_dataset('meta', data=meta, compression=8)
            f.create_dataset('positions', data=locs, compression=8)
            f.create_dataset('images', data=imgs, compression=8)
            f.create_dataset('phos', data=phos, compression=8)

    def save_truths(self):
        ext = ['', '__median'][int(self.is_med)]
        fname = os.path.join(self.save_dir, \
            f'truths__{self.session}__{self.run}{ext}.h5')

        with h5py.File(fname, 'w') as f:
            f.create_dataset('true_position', data=self.true_position)

    def load_truths(self):
        ext = ['', '__median'][int(self.is_med)]
        fname = os.path.join(self.save_dir, \
            f'truths__{self.session}__{self.run}{ext}.h5')

        with h5py.File(fname, 'r') as f:
            self.true_position = f['true_position'][()]

############################################
############################################

############################################
####    Pupil Mask ####
############################################

    def load_pupil_mask(self, mask_type):

        #Use spider mask or just round aperture
        if mask_type == 'spiders':
            #Load Pupil Mask
            with h5py.File(os.path.join(self.xtras_dir,'pupil_mask.h5'), 'r') as f:
                full_mask = f['mask'][()]

            #Do affine transform
            scaling = full_mask.shape[0]/self.num_pts
            dx = -scaling*(self.img_shape[0] - self.num_pts)/2
            affmat = np.array([[scaling, 0, dx], [0, scaling, dx]])
            pupil_mask = affine_transform(full_mask, affmat, \
                output_shape=self.img_shape, order=5)

            #Make binary
            pupil_mask[pupil_mask <  0.5] = 0
            pupil_mask[pupil_mask >= 0.5] = 1

        elif mask_type == 'round':

            #Get indices of outside aperture
            rr = np.hypot(*(np.indices(self.img_shape) - self.img_shape[0]/2))
            #Just round aperture
            pupil_mask = np.ones(self.img_shape)
            pupil_mask[rr > self.num_pts/2] = 0.

        else:

            #No mask
            pupil_mask = np.ones(self.img_shape)

        #Build spiders mask (boolean where mask is 0)
        spiders = pupil_mask == 0

        #Cleanup
        del pupil_mask

        return spiders

############################################
############################################
