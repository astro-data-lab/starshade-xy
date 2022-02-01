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
            'data_dir':         '',
            'xtras_dir':        '../../quadrature_code/xtras',
            'save_dir':         'Processed_Images',
            'run':              '',
            'session':          '',
            'do_round_mask':    False,
            ### Saving ###
            'do_save':          False,
            'do_plot':          False,
            ### Observation ###
            'is_median':        False,
            'base_num_pts':     96,
            'image_pad':        10,
            'num_pixel_base':   250,
            ### Truth Sensor ###
            'image_center':     None,
            'sensing_method':   'model',        #Options: ['model', 'centroid']
            'cen_threshold':    0.75,           #Centroiding threshold
            'physical_rad':     2.5e-3,         #Radius of physical aperture
            'wave':             403e-9,
            'ss_radius':        10.5e-3*np.sqrt(680/638),
            'z1':               50.,
            'ccd_dark':         7e-4,           #Dark noise [e/px/s]
            'ccd_read':         4.78,           #Read noise [e/px/frame]
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
        self.load_dir = os.path.join(self.data_dir, self.session, self.run)

        #FIXED
        pupil_mag = 1.764
        pixel_size = 13e-6

        #Derived
        self.tel_diameter = self.base_num_pts * pupil_mag*pixel_size
        self.physical_rad_px = self.physical_rad / (pupil_mag*pixel_size)

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

        #Load image record
        self.record = np.genfromtxt(os.path.join(self.load_dir, 'record.csv'), delimiter=',')

        #Prepare true position
        self.num_imgs = len(self.record)
        self.true_position = np.empty((self.num_imgs,2))
        self.true_amplitude = np.zeros(self.num_imgs)

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
        amps = np.empty((0,))
        locs = np.empty((0, 2))
        meta = np.empty((0, 3))
        for i in range(self.num_imgs):

            #Get image
            img, exp = self.get_image(int(self.record[i][0]))

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
            if self.is_median:
                nframe = img.shape[0]
                img = np.array([np.median(img, 0)])
            else:
                nframe = 1

            #Get current position
            if mask_type == 'none':
                #Position guess (flip x-sign b/c optics)
                pos0 = self.record[i][1:] * np.array([-1, 1])
                #Get true position
                pos, amp = self.truth.get_position(img, exp, pos0=pos0)
                #Save
                self.true_position[i] = pos
                self.true_amplitude[i] = amp
            else:
                #Get stored true position
                pos = self.true_position[i]
                amp = self.true_amplitude[i]

            #Plot
            if self.do_plot:
                import matplotlib.pyplot as plt;plt.ion()
                print(i, img.max()/amp)
                img_extent = [self.truth.xx[0]*self.truth.pupil_mag, \
                    self.truth.xx[-1]*self.truth.pupil_mag, \
                    self.truth.yy[-1]*self.truth.pupil_mag, \
                    self.truth.yy[0]*self.truth.pupil_mag]

                plt.cla()
                plt.imshow(img[0], extent=img_extent)
                plt.plot(0, 0, 'kx')
                plt.plot(0, 0, 'k+')
                plt.plot(pos[0], pos[1], 'ro')
                breakpoint()

            #Store images + positions
            imgs = np.concatenate((imgs, img))
            amps = np.concatenate((amps, [amp]))
            locs = np.concatenate((locs, [pos]*img.shape[0]))
            #Store exposure time + backgrounds + number of frames
            meta = np.concatenate((meta, [[exp, back, nframe]]*img.shape[0]))

        #Trim images
        if mask_type != 'none':
            imgs = image_util.crop_image(imgs, None, self.num_pts//2+self.image_pad)

        #Save Data
        if self.do_save:
            self.save_data(mask_type, imgs, amps, locs, meta)
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

    def get_image(self, inum):
        #Get data
        fname = os.path.join(self.load_dir, f'image__{str(inum).zfill(4)}.fits')
        img, exp = pfunc.get_image_data(fname)

        return img, exp

    def save_data(self, mask_type, imgs, amps, locs, meta):

        ext = ['', '__median'][int(self.is_median)]
        fname = os.path.join(self.save_dir, \
            f'{self.session}__{self.run}__{mask_type}{ext}.h5')

        with h5py.File(fname, 'w') as f:
            f.create_dataset('num_tel_pts', data=self.num_pts)
            f.create_dataset('base_num_pts', data=self.base_num_pts)
            f.create_dataset('binning', data=self.binning)
            f.create_dataset('image_pad', data=self.image_pad)
            f.create_dataset('tel_diameter', data=self.tel_diameter)
            f.create_dataset('meta', data=meta, compression=8)
            f.create_dataset('positions', data=locs, compression=8)
            f.create_dataset('amplitudes', data=amps, compression=8)
            f.create_dataset('images', data=imgs, compression=8)

    def save_truths(self):
        ext = ['', '__median'][int(self.is_median)]
        fname = os.path.join(self.save_dir, \
            f'truths__{self.session}__{self.run}{ext}.h5')

        with h5py.File(fname, 'w') as f:
            f.create_dataset('true_position', data=self.true_position)
            f.create_dataset('true_amplitude', data=self.true_amplitude)

    def load_truths(self):
        ext = ['', '__median'][int(self.is_median)]
        fname = os.path.join(self.save_dir, \
            f'truths__{self.session}__{self.run}{ext}.h5')

        with h5py.File(fname, 'r') as f:
            self.true_position = f['true_position'][()]
            self.true_amplitude = f['true_amplitude'][()]

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
