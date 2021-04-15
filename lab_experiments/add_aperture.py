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
from scipy.ndimage import affine_transform

session = ['new_data_30s', 'data_30s_bin2', 'data_20s_bin4', 'data_60s_bin4'][3]

is_med = True

#Choose aperture size for proper scaling to space
num_pts = 96
image_pad = 10

#aperture size (from pupil magnification)
Dtel = num_pts * 1.748*13e-6

#TODO: hide
data_dir = f'/home/aharness/Research/Frick_Lab/Data/FFNN/{session}'

#Read record
record = np.genfromtxt(f'{data_dir}/record.csv', delimiter=',')

def get_image(inum):
    with fits.open(f'{data_dir}/image__{str(inum).zfill(4)}.fits') as hdu:
        data = hdu[0].data.astype(float)
    return data

#Get image shape
img0 = get_image(len(record)//2)
img_shp = img0.shape[-2:]

#Get binning
nbin = int(np.round(250/img_shp[0]))
num_pts //= nbin

#Load Pupil Mask
with h5py.File(f'../diffraction_code/xtras/pupil_mask.h5', 'r') as f:
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

#Tile for multiple exposures
spiders = np.tile(spiders, (img0.shape[0], 1, 1))
out_mask = np.tile(out_mask, (img0.shape[0], 1, 1))

#Loop through steps and get images
imgs = np.empty((0,) + img_shp)
locs = np.empty((0, 2))
for i in range(len(record)):

    #Current position
    pos = record[i][1:]

    #Get image
    img = get_image(int(record[i][0]))

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
    img -= np.median(excess)

    #Take median
    if is_med:
        img = np.array([np.median(img, 0)])

    #Plot
    if [False, True][0]:
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

#Trim images
imgs = image_util.crop_image(imgs, None, num_pts//2+image_pad)

plt.cla()
plt.imshow(imgs[len(imgs)//2])

#Save
if [False, True][1]:

    ext = ['', '__median'][int(is_med)]

    with h5py.File(f'./Results/{session}{ext}.h5', 'w') as f:
        f.create_dataset('num_tel_pts', data=num_pts)
        f.create_dataset('image_pad', data=image_pad)
        f.create_dataset('tel_diameter', data=Dtel)
        f.create_dataset('positions', data=locs, compression=8)
        f.create_dataset('images', data=imgs, compression=8)

breakpoint()
