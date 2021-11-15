import numpy as np
import matplotlib.pyplot as plt;plt.ion()
import h5py
import image_util
from scipy.ndimage import shift

run1 = 'run__11_15_21_b__data_5s_bin1'
run2 = 'run__6_01_21__data_2s_bin1'

ap = ['none', 'spiders'][0]
pos0 = np.array([0., 0.])

pupil_mag = 1.764
pixel_size = 13e-6

#Get closest to center for the run with fewest images
with h5py.File(f'./Processed_Images/{run1}__{ap}__median.h5', 'r') as f:
    pos = f['positions'][()]
    ind1 = np.argmin(np.hypot(pos[:,0] - pos0[0], pos[:,1] - pos0[1]))
    img1 = f['images'][ind1]
    pos1 = pos[ind1]

#Find closest image to that point
with h5py.File(f'./Processed_Images/{run2}__{ap}__median.h5', 'r') as f:
    pos = f['positions'][()]
    ind2 = np.argmin(np.hypot(pos[:,0] - pos1[0], pos[:,1] - pos1[1]))
    img2 = f['images'][ind2]
    pos2 = pos[ind2]

# img1 = shift(img1, ((pos1-pos2)/pupil_mag/pixel_size)[::-1], order=5)

if img2.shape[-1] > img1.shape[-1]:
    img2 = image_util.crop_image(img2, None, img1.shape[0]//2)

else:
    img1 = image_util.crop_image(img1, None, img2.shape[0]//2)

plt.figure()
plt.imshow(img1)
plt.figure()
plt.imshow(img2)

plt.figure()
plt.colorbar(plt.imshow((img1 - img2)/0.03*100))

plt.figure()
plt.colorbar(plt.imshow(abs(img1 - img2)/0.03*100))

breakpoint()
