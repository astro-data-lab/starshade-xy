"""
compare_model_lab.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 06-11-2021

Description: Script to compare experimental and model pupil plane images.

"""

import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
import imp
from mpl_toolkits.axes_grid1 import ImageGrid

#Import diffraq
diff_dir = os.path.join(os.pardir, os.pardir, "quadrature_code")
diffraq = imp.load_source("diffraq", os.path.join(diff_dir,"diffraq","__init__.py"))
import image_util

session = 'run__6_01_21'
run = 'data_2s_bin1'
mask_type = ['spiders', 'round', 'none'][0]
is_med = True

#Desired center (in fractions of tel_radius)
# frac_cen = [0.65, -0.65] #1,2,37
# frac_cen = [-0.15, 0.35] #1
frac_cen = [1, 0] #1

do_save = [False, True][0]

if not do_save:
    plt.ion()

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

# breakpoint()

#Add one pixel to match diffraq
# pos0 += lab_params['tel_diameter']/lab_params['num_tel_pts']
# pos0[0] -= lab_params['tel_diameter']/lab_params['num_tel_pts']

# dx = lab_params['tel_diameter']/lab_params['num_tel_pts']

# pos0[0] += dx
# pos0[1] += dx*.25

############################################
####    Simulate Image ####
############################################

#Specify simulation parameters
params = {

    ### Lab ###
    'wave':             0.403e-6,

    ### Telescope ###
    'with_spiders':     mask_type == 'spiders',
    'skip_mask':        mask_type == 'none',
    'tel_shift':        pos0,
    # 'wfe_modes':        [(1,1,-5e-7*0), (0,2,-5e-7)],
    # 'pupil_mag':        0.575,

    ### Starshade ###
    'apod_name':        'lab_ss',
    # 'circle_rad':       25.054e-3,
    'num_petals':       12,

    ### Saving ###
    'do_save':          False,
    'verbose':          False,
    'xtras_dir':        os.path.join(diff_dir,"xtras"),

}

for k in lab_keys:
    params[k] = lab_params[k]

#Load simulator class
sim = diffraq.Simulator(params)
sim.setup_sim()

#Get diffraction and convert to intensity
sim_img = np.abs(sim.calculate_diffraction())**2


# plt.imshow(sim_img)
# breakpoint()
# sim_pup = sim.calculate_diffraction()
# sim_img, grid_pts = sim.focuser.calculate_image(sim_pup)

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

# #Residuals
# res1 = lab_img/lab_img.max() - sim_img/sim_img.max()
res2 = lab_img/lab_img.mean() - sim_img/sim_img.mean()

#Get sum of center peak
dn = 15
cen = (-pos0 / (sim.tel_pts[1] - sim.tel_pts[0]) + np.array(lab_img.shape)/2).astype(int)
lab_sum = image_util.crop_image(lab_img, cen, dn).sum()
sim_sum = image_util.crop_image(sim_img, cen, dn).sum()

# lab_sum = 0.8226
# sim_sum = 3.32e-3 * 1.4
# lab_sum = lab_img.max()
# sim_sum = sim_img.max()


lab_img /= lab_sum
sim_img /= sim_sum

sim_img /= lab_img.max()
lab_img /= lab_img.max()

# res1 = lab_img/lab_sum - sim_img/sim_sum
res1 = (lab_img - sim_img)*100

#Round apertures
if mask_type == 'none':
    lab_img = image_util.round_aperture(lab_img, dr=15)
    sim_img = image_util.round_aperture(sim_img, dr=15)
    res1 = image_util.round_aperture(res1, dr=15)
    res2 = image_util.round_aperture(res2, dr=15)

#Plot
if [False, True][0]:
    fig, axes = plt.subplots(1, 2, figsize=(7, 5), sharex=True, sharey=True)
    lp = axes[0].imshow(lab_img)
    sp = axes[1].imshow(sim_img)
    axes[0].set_title('Lab')
    axes[1].set_title('Sim')
    plt.colorbar(lp, ax=axes[0], orientation='horizontal')
    plt.colorbar(sp, ax=axes[1], orientation='horizontal')

fig = plt.figure(1, figsize=(9,9))
grid = ImageGrid(fig, 111,
            nrows_ncols=(2, 2),
            axes_pad=(0.5,1.)[::-1],
            label_mode="L",
            share_all=True,
            cbar_location="right",
            cbar_mode="each",
            cbar_size="5%",
            cbar_pad=0.05,
            )

for i in range(4):
    grid[i].axis('off')
    grid[i].plot(lab_img.shape[-1]/2, lab_img.shape[-1]/2, 'k+')

grid[0].set_title('Lab')
grid[1].set_title('Sim')
grid[2].set_title('Lab - Sim')
grid[3].set_title('|Lab - Sim|')

ll = grid[0].imshow(lab_img, vmax=1)
mm = grid[1].imshow(sim_img, vmax=1)
ff = grid[2].imshow(res1)
aa = grid[3].imshow(abs(res1))

orien = 'vertical'
cbar1 = plt.colorbar(ll, cax=grid[0].cax, orientation=orien, label=f'Normalized')
cbar2 = plt.colorbar(mm, cax=grid[1].cax, orientation=orien, label=f'Normalized')
cbar3 = plt.colorbar(ff, cax=grid[2].cax, orientation=orien, label=f'% Error')
cbar4 = plt.colorbar(aa, cax=grid[3].cax, orientation=orien, label=f'% Error')


plt.figure()
plt.plot(lab_img[lab_img.shape[-1]//2])
plt.plot(sim_img[sim_img.shape[-1]//2])

if do_save:
    fig.savefig(f'./Plots/x{frac_cen[0]*10:.0f}_y{frac_cen[1]*10:.0f}__{mask_type}.png')
else:
    breakpoint()
