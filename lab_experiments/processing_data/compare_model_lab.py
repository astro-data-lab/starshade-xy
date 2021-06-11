"""
compare_model_lab.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 06-11-2021

Description: Script to compare experimental and model pupil plane images.

"""

import numpy as np
import matplotlib.pyplot as plt;plt.ion()
import h5py
import os
import imp
#Import BDW
diff_dir = os.path.join(os.pardir, os.pardir, "diffraction_code")
imp.load_source("bdw", os.path.join(diff_dir,"bdw.py"))
from bdw import BDW
from scipy.ndimage import shift
import image_util

session = 'run__6_01_21'
run = 'data_2s_bin1'
mask_type = ['spiders', 'round', 'none'][2]
is_med = True

#Desired center (in fractions of tel_radius)
frac_cen = [0., 0.5]

############################################
####    Load Lab Data ####
############################################

lab_keys = ['num_tel_pts', 'tel_diameter', 'image_pad']
lab_params = {}

#Get truth positions
tru_name = f'{session}__{run}__none{["", "__median"][int(is_med)]}'
with h5py.File(f'./Truth_Results/{tru_name}__truth_model.h5', 'r') as f:
    tru_pos = f['truth'][()]

#Load lab data
lab_name = f'{session}__{run}__{mask_type}{["", "__median"][int(is_med)]}'
with h5py.File(f'./Results/{lab_name}.h5', 'r') as f:
    #get center
    cen = np.array(frac_cen) * f['tel_diameter']/2

    #Find image closest to specified point
    ind0 = np.argmin(np.hypot(*(tru_pos - cen).T))
    pos0 = tru_pos[ind0]

    #Get on-axis image
    lab_img = f['images'][ind0]

    #Get params
    for k in lab_keys:
        lab_params[k] = f[k][()]

#Overwrite params if no mask
if mask_type == 'none':
    lab_params['tel_diameter'] *= lab_img.shape[-1]/lab_params['num_tel_pts']
    lab_params['num_tel_pts'] = lab_img.shape[-1]
    lab_params['image_pad'] = 0

############################################
####    Simulate Image ####
############################################

#Specify simulation parameters
params = {

    ### Lab ###
    'wave':             0.405e-6,

    ### Telescope ###
    'with_spiders':     mask_type == 'spiders',
    'skip_mask':        mask_type == 'none',
    'tel_shift':        pos0,

    ### Starshade ###
    'apod_name':        'lab_ss',
    'num_petals':       12,

    ### Saving ###
    'do_save':          False,
    'verbose':          False,
    'xtras_dir':        os.path.join(diff_dir,"xtras"),

}

for k in lab_keys:
    params[k] = lab_params[k]

#Load BDW class
bdwc = BDW(params)

#Check if file exists
sim_file = f'./tmp_save/bdw_image__{mask_type}__{str(ind0).zfill(4)}.h5'

#Calculate or Load simulated image?
if [False, True][0] or not os.path.exists(sim_file):

    print('Running BDW...')

    #Get diffraction and convert to intensity
    sim_img = np.abs(bdwc.calculate_diffraction())**2

    #Save image
    with h5py.File(sim_file, 'w') as f:
        f.create_dataset('image', data=sim_img)

else:

    #Load image
    with h5py.File(sim_file, 'r') as f:
        sim_img = f['image'][()]

#Pad image if needed
if sim_img.shape != lab_img.shape:
    sim_img = image_util.pad_array(sim_img, NN=lab_img.shape[-1])

############################################
####    Compare Images ####
############################################

#Normalize
# lab_img /= lab_img.max()
# sim_img /= sim_img.max()

# lab_img /= np.mean(lab_img)
# sim_img /= np.mean(sim_img)

# lab_img = shift(lab_img, (1, 1.), order=5)

# #Residuals
# res1 = lab_img/lab_img.max() - sim_img/sim_img.max()
res2 = lab_img/lab_img.mean() - sim_img/sim_img.mean()

#Get sum of center peak
dn = 10
cen = (-pos0 / (bdwc.tel_pts[1] - bdwc.tel_pts[0]) + np.array(lab_img.shape)/2).astype(int)
lab_sum = image_util.crop_image(lab_img, cen, dn).sum()
sim_sum = image_util.crop_image(sim_img, cen, dn).sum()

lab_sum = 0.0041839807708278515
sim_sum = 1.0936061209758707

res1 = lab_img/lab_sum - sim_img/sim_sum

#Round apertures
res1 = image_util.round_aperture(res1, dr=10)
res2 = image_util.round_aperture(res2, dr=10)

#convert to percentages
res1 *= 100 / (lab_img.max() / lab_sum)
res2 *= 100 / (lab_img.max() / lab_sum)

#Plot
fig, axes = plt.subplots(1, 2, figsize=(7, 5), sharex=True, sharey=True)
lp = axes[0].imshow(lab_img)
sp = axes[1].imshow(sim_img)
axes[0].set_title('Lab')
axes[1].set_title('Sim')
plt.colorbar(lp, ax=axes[0], orientation='horizontal')
plt.colorbar(sp, ax=axes[1], orientation='horizontal')

rfig, raxes = plt.subplots(1, 2, figsize=(7, 5), sharex=True, sharey=True)
fres = raxes[0].imshow(res1)
ares = raxes[1].imshow(abs(res1))
raxes[0].set_title('Lab - Sim')# (peak norm)')
raxes[1].set_title('|Lab - Sim|')#' (peak norm)')
plt.colorbar(fres, ax=raxes[0], orientation='horizontal', label=f'% scaled by peak')
plt.colorbar(ares, ax=raxes[1], orientation='horizontal', label=f'% scaled by peak')


# rfig, raxes = plt.subplots(1, 2, figsize=(7, 5), sharex=True, sharey=True)
# fres = raxes[0].imshow(res2)
# ares = raxes[1].imshow(abs(res2))
# raxes[0].set_title('Lab - Sim (mean norm)')
# raxes[1].set_title('|Lab - Sim| (mean norm)')
# plt.colorbar(fres, ax=raxes[0], orientation='horizontal')
# plt.colorbar(ares, ax=raxes[1], orientation='horizontal')



breakpoint()
