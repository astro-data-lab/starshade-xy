"""
experiment_image_processor.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 06-28-2021

Description: Script to add spiders and secondary to experimental images.

"""

import numpy as np
import matplotlib.pyplot as plt;plt.ion()
from astropy.io import fits
import h5py
import glob
import image_util
import photo_functions as pfunc
from scipy.ndimage import affine_transform

class Experiment_Image_Processor(object):

    def __init__(self, params={}):
        self.set_parameters(params)

############################################
####    Initialization ####
############################################

    def set_parameters(self, params):
        #Default parameters
        def_pms = {
            ### Loading ###
            'base_dir':         '/home/aharness/Research/Frick_Lab/Data/FFNN',
            'xtras_dir':        '../../quadrature_code/xtras',
            'save_dir':         './Results',
            'all_runs':         [],
            'session':          '',
            'is_med':           False,
            'mask_type':        'none',
            ### Saving ###
            'do_save':          False,
            'do_plot':          False,
            ### Observation ###
            'base_num_pts':     96,
            'image_pad':        10,
            'binning':          1,
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
        self.load_dir = f'{self.base_dir}/{self.session}'

        #Derived
        self.Dtel = self.base_num_pts * 1.764*13e-6
        self.num_pts = self.base_num_pts // self.binning
        self.do_crop = self.mask_type != 'none'

############################################
############################################

############################################
####    Main Script ####
############################################

    def run_script(self):

        #Load calibration
        self.load_calibration()

        #Peek at image data
        self.peek_at_image()

        #Load pupil mask
        self.load_pupil_mask()

        #Loop through and process multiple runs
        for run in self.all_runs:

            #Process images
            imgs, locs, meta = self.process_images(run)

            #Save data
            if self.do_save:
                self.save_data(run, imgs, locs, meta)

            # breakpoint()

    ############################################

    def process_images(self, run):

        #Load image record
        record = np.genfromtxt(f'{self.load_dir}/{run}/record.csv', delimiter=',')

        #Loop through steps and get images and exposure times + backgrounds
        imgs = np.empty((0,) + self.img_shape)
        locs = np.empty((0, 2))
        meta = np.empty((0, 3))
        for i in range(len(record)):

            #Current position
            pos = record[i][1:]

            #Get image
            img, exp = self.get_image(run, int(record[i][0]))

            #Get excess regions
            nbk = 40//self.binning
            excess = np.concatenate((img[:,:nbk,:nbk], img[:,:nbk,-nbk:], \
                img[:,-nbk:,:nbk], img[:,-nbk:,-nbk:])).flatten()

            #Add mask
            img *= self.pupil_mask

            #Add noise in spiders and outside aperture
            img[self.spiders] = np.random.choice(excess, (img.shape[0], self.nspid)).flatten()
            img[self.out_mask] = np.random.choice(excess, (img.shape[0], self.nout)).flatten()

            #Subtract background
            back = np.median(excess)
            img -= back

            #Take median
            if self.is_med:
                nframe = img.shape[0]
                img = np.array([np.median(img, 0)])
            else:
                nframe = 1

            #Plot
            if self.do_plot:
                print(i, pos*1e3)
                for j in range(img.shape[0]):
                    plt.cla()
                    cimg = image_util.crop_image(img[j], None, self.num_pts//2+self.image_pad)
                    plt.imshow(cimg)
                    print(self.num_pts, cimg.shape)
                    breakpoint()

            #Store images + positions
            imgs = np.concatenate((imgs, img))
            locs = np.concatenate((locs, [pos]*img.shape[0]))
            #Store exposure time + backgrounds + number of frames
            meta = np.concatenate((meta, [[exp, back, nframe]]*img.shape[0]))

        #Trim images
        if self.do_crop:
            imgs = image_util.crop_image(imgs, None, self.num_pts//2+self.image_pad)

        #Flip sign of x-coord b/c optics
        locs[:,0] *= -1

        return imgs, locs, meta

############################################

############################################
####    Misc Functions ####
############################################

    def peek_at_image(self):
        img, exp = self.get_image(self.all_runs[0], 1)
        self.num_kin = img.shape[0]
        self.img_shape = img.shape[1:]

    def load_calibration(self):

        #Get photometer data
        self.photo_data = pfunc.load_photometer_data(f'{self.load_dir}', None)

        #Load calibration data
        fname = f'{self.load_dir}/cal_con.fits'
        cimg, cexp, cpho = pfunc.get_image_data(fname, self.photo_data)

        #Crop
        max_pt = np.unravel_index(np.argmax(cimg[0]), cimg.shape[1:])[::-1]
        cimg = image_util.crop_image(cimg, max_pt, 50)

        #Get normalized median
        med = np.median(cimg / cpho[:,None,None], 0)

        #Get peak from max
        self.cal_value = med.max()

    def get_image(self, run, inum):
        #Get data
        fname = f'{self.load_dir}/{run}/image__{str(inum).zfill(4)}.fits'
        img, exp, pho = pfunc.get_image_data(fname, self.photo_data)
        #Normalize
        img /= pho[:,None,None] * self.cal_value
        return img, exp

    def save_data(self, run, imgs, locs, meta):

        ext = ['', '__median'][int(self.is_med)]

        with h5py.File(f'{self.save_dir}/{self.session}__{run}__{self.mask_type}{ext}.h5', 'w') as f:
            f.create_dataset('cal_value', data=self.cal_value)
            f.create_dataset('num_tel_pts', data=self.num_pts)
            f.create_dataset('base_num_pts', data=self.base_num_pts)
            f.create_dataset('binning', data=self.binning)
            f.create_dataset('image_pad', data=self.image_pad)
            f.create_dataset('tel_diameter', data=self.Dtel)
            f.create_dataset('meta', data=meta, compression=8)
            f.create_dataset('positions', data=locs, compression=8)
            f.create_dataset('images', data=imgs, compression=8)

############################################
############################################

############################################
####    Pupil Mask ####
############################################

    def load_pupil_mask(self):

        #Use spider mask or just round aperture
        if self.mask_type == 'spiders':
            #Load Pupil Mask
            with h5py.File(f'{self.xtras_dir}/pupil_mask.h5', 'r') as f:
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

            #Get indices of spiders and outside aperture
            rr = np.hypot(*(np.indices(self.img_shape) - self.img_shape[0]/2))
            spiders = (rr <= self.num_pts/2) & (pupil_mask == 0)
            out_mask = rr > self.num_pts/2
            nspid = np.count_nonzero(spiders)
            nout = np.count_nonzero(out_mask)

            del full_mask

        elif self.mask_type == 'round':

            #Get indices of outside aperture
            rr = np.hypot(*(np.indices(self.img_shape) - self.img_shape[0]/2))
            out_mask = rr > self.num_pts/2
            nout = np.count_nonzero(out_mask)
            #No spiders
            spiders = np.zeros(self.img_shape).astype(bool)
            nspid = np.count_nonzero(spiders)

            #Just round aperture
            pupil_mask = np.ones(self.img_shape)
            pupil_mask[out_mask] = 0.

        else:

            #No mask
            nout, nspid = 0, 0
            pupil_mask = np.ones(self.img_shape)

            out_mask = np.zeros(self.img_shape).astype(bool)
            spiders = np.zeros(self.img_shape).astype(bool)

        #Tile for multiple exposures
        spiders = np.tile(spiders, (self.num_kin, 1, 1))
        out_mask = np.tile(out_mask, (self.num_kin, 1, 1))

        #Store
        self.pupil_mask = pupil_mask
        self.spiders = spiders
        self.out_mask = out_mask
        self.nspid = nspid
        self.nout = nout

############################################
############################################
