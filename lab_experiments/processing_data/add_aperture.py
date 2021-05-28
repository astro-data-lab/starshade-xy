"""
add_aperture.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 04-05-2021

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

###########################################

all_runs = ['data_1s_bin1', 'data_1s_bin2', 'data_1s_bin4']
session = 'run__5_26_21'

is_med = False
do_save = False

#Mask type. Use 'none' for centroiding images
mask_type = ['spiders', 'round', 'none'][0]

do_plot = [False, True][0]

#Base directory
base_dir = '/home/aharness/Research/Frick_Lab/Data/FFNN'

###########################################

#Get photometer data
photo_data = pfunc.load_photometer_data(f'{base_dir}/{session}', None)

###########################################

#Load calibration data
fname = f'{base_dir}/{session}/cal_con.fits'
cimg, cexp, cpho = pfunc.get_image_data(fname, photo_data)

#Crop
max_pt = np.unravel_index(np.argmax(cimg[0]), cimg.shape[1:])[::-1]
cimg = image_util.crop_image(cimg, max_pt, 50)

#Get normalized median
med = np.median(cimg / cpho[:,None,None], 0)

#Get peak from max
cal_value = med.max()

###########################################

#Loop through and process multiple sessions
for run in all_runs:

    #Don't crop if no mask
    do_crop = mask_type != 'none'

    #Choose aperture size for proper scaling to space
    base_num_pts = 96
    image_pad = 10

    #Start with base num pts    (to be changed later by binning)
    num_pts = base_num_pts

    #aperture size (from pupil magnification)
    Dtel = num_pts * 1.748*13e-6

    #Read record
    data_dir = f'{base_dir}/{session}/{run}'
    record = np.genfromtxt(f'{data_dir}/record.csv', delimiter=',')

    def get_image(inum):
        #Get data
        fname = f'{data_dir}/image__{str(inum).zfill(4)}.fits'
        img, exp, pho = pfunc.get_image_data(fname, photo_data)
        #Normalize
        img /= pho[:,None,None] * cal_value
        return img, exp

    #Get image shape
    img0, exp0 = get_image(len(record)//2)
    img_shp = img0.shape[-2:]

    #Get binning
    nbin = int(np.round(250/img_shp[0]))
    num_pts //= nbin

    ##################

    #Use spider mask or just round aperture
    if mask_type == 'spiders':
        #Load Pupil Mask
        with h5py.File(f'../../diffraction_code/xtras/pupil_mask.h5', 'r') as f:
            full_mask = f['mask'][()]

        #Do affine transform
        scaling = full_mask.shape[0]/num_pts
        dx = -scaling*(img_shp[0]-num_pts)/2
        affmat = np.array([[scaling, 0, dx], [0, scaling, dx]])
        pupil_mask = affine_transform(full_mask, affmat, output_shape=img_shp, order=5)

        #Make binary
        pupil_mask[pupil_mask <  0.5] = 0
        pupil_mask[pupil_mask >= 0.5] = 1

        #Get indices of spiders and outside aperture
        rr = np.hypot(*(np.indices(img_shp) - img_shp[0]/2))
        spiders = (rr <= num_pts/2) & (pupil_mask == 0)
        out_mask = rr > num_pts/2
        nspid = np.count_nonzero(spiders)
        nout = np.count_nonzero(out_mask)

    elif mask_type == 'round':

        #Get indices of outside aperture
        rr = np.hypot(*(np.indices(img_shp) - img_shp[0]/2))
        out_mask = rr > num_pts/2
        nout = np.count_nonzero(out_mask)
        #No spiders
        spiders = np.zeros(img_shp).astype(bool)
        nspid = np.count_nonzero(spiders)

        #Just round aperture
        pupil_mask = np.ones(img_shp)
        pupil_mask[out_mask] = 0.

    else:

        #No mask
        nout, nspid = 0, 0
        pupil_mask = np.ones(img_shp)

        out_mask = np.zeros(img_shp).astype(bool)
        spiders = np.zeros(img_shp).astype(bool)

    #Tile for multiple exposures
    spiders = np.tile(spiders, (img0.shape[0], 1, 1))
    out_mask = np.tile(out_mask, (img0.shape[0], 1, 1))

    ##################

    #Loop through steps and get images and exposure times + backgrounds
    imgs = np.empty((0,) + img_shp)
    locs = np.empty((0, 2))
    meta = np.empty((0, 3))
    for i in range(len(record)):

        #Current position
        pos = record[i][1:]

        #Get image
        img, exp = get_image(int(record[i][0]))

        #Get excess regions
        nbk = 40//nbin
        excess = np.concatenate((img[:,:nbk,:nbk], img[:,:nbk,-nbk:], \
            img[:,-nbk:,:nbk], img[:,-nbk:,-nbk:])).flatten()

        #Add mask
        img *= pupil_mask

        #Add noise in spiders and outside aperture
        img[spiders] = np.random.choice(excess, (img.shape[0], nspid)).flatten()
        img[out_mask] = np.random.choice(excess, (img.shape[0], nout)).flatten()

        #Subtract background
        back = np.median(excess)
        img -= back

        #Take median
        if is_med:
            nframe = img.shape[0]
            img = np.array([np.median(img, 0)])
        else:
            nframe = 1

        #Plot
        if do_plot:
            print(i, pos*1e3)
            for j in range(img.shape[0]):
                plt.cla()
                # plt.imshow(img[j])
                cimg = image_util.crop_image(img[j], None, num_pts//2+image_pad)
                plt.imshow(cimg)
                print(num_pts, cimg.shape)
                breakpoint()

        #Store images + positions
        imgs = np.concatenate((imgs, img))
        locs = np.concatenate((locs, [pos]*img.shape[0]))
        #Store exposure time + backgrounds + number of frames
        meta = np.concatenate((meta, [[exp, back, nframe]]*img.shape[0]))

    ##################

    #Trim images
    if do_crop:
        imgs = image_util.crop_image(imgs, None, num_pts//2+image_pad)

    #Flip sign of x-coord b/c optics
    locs[:,0] *= -1

    plt.cla()
    plt.imshow(imgs[len(imgs)//2])

    #Save
    if do_save:

        ext = ['', '__median'][int(is_med)]

        with h5py.File(f'./Results/{session}__{run}__{mask_type}{ext}.h5', 'w') as f:
            f.create_dataset('num_tel_pts', data=num_pts)
            f.create_dataset('base_num_pts', data=base_num_pts)
            f.create_dataset('binning', data=nbin)
            f.create_dataset('image_pad', data=image_pad)
            f.create_dataset('tel_diameter', data=Dtel)
            f.create_dataset('meta', data=meta, compression=8)
            f.create_dataset('positions', data=locs, compression=8)
            f.create_dataset('images', data=imgs, compression=8)

    # breakpoint()
