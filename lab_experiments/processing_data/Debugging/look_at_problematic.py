import numpy as np
import matplotlib.pyplot as plt;plt.ion()
import glob
from comparison_funcs import *
from mpl_toolkits.axes_grid1 import ImageGrid


mask_type = ['spiders', 'round', 'none'][0]

#############

#Session + run the corresponds to Problematic images
sess_run = f'run__5_26_21__data_1s_bin1'
is_med = False

#Get problematic image filenames
fnames = np.array(glob.glob('./Problematic/bad*.npy'))

#Extract image numbers
nums = np.array([int(fn.split('bad')[-1].split('.npy')[0]) for fn in fnames])

#Sort by image number
fnames = fnames[np.argsort(nums)]
nums = nums[np.argsort(nums)]

#Plot stuff
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
labels = ['Lab', 'Sim', 'Lab - Sim', '|Lab - Sim|']
orien = 'vertical'
cbar_labs = ['Normalized']*2 + [f'% Error']*2

#Build colorbars
cbars = []
for i in range(4):
    cbars.append(plt.colorbar(grid[i].imshow(np.zeros((74,74))), \
        cax=grid[i].cax, orientation=orien, label=cbar_labs[i]))

#Loop through images
for i in range(len(nums))[::10]:

    #Image number
    num = nums[i]

    #Load image
    lab_img, lab_pms = load_lab_data(num, sess_run, mask_type, is_med)

    #Simulate image
    mod_img, sim = simulate_image(lab_pms, mask_type)

    #Center pixel
    cen = -np.array(sim.tel_shift) / (sim.tel_pts[1] - sim.tel_pts[0])

    #Normalize
    lab_img, mod_img = normalize_images(lab_img, mod_img, cen, False)

    #Residual
    res = (lab_img - mod_img)/lab_img.max()*100

    #Plot
    print(num, abs(res).max())

    imgs = [lab_img, mod_img, res, abs(res)]

    for i in range(4):
        #Plot
        out = grid[i].imshow(imgs[i])
        #Update Colorbar
        cbars[i].update_normal(out)
        #Cleanup
        grid[i].axis('off')
        grid[i].plot(lab_img.shape[-1]/2, lab_img.shape[-1]/2, 'k+')
        grid[i].set_title(labels[i])


    breakpoint()
