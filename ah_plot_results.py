import numpy as np
import matplotlib.pyplot as plt
import h5py
import os
from mpl_toolkits.axes_grid1 import ImageGrid

data_run = 'run__6_01_21__data_1s_bin1__spiders__median'

model = 'New'

do_save = [False, True][0]

results_dir = 'Results'
plot_dir = 'Plots'

#Load results
with h5py.File(os.path.join(results_dir, f'{data_run}__{model}.h5'), 'r') as f:
    xerr = f['xerr'][()]
    yerr = f['yerr'][()]
    positions = f['positions'][()]

rerr = np.hypot(xerr, yerr)

if not do_save:
    plt.ion()

#Error Map
fig = plt.figure(1, figsize=(11,5))
grid = ImageGrid(fig, 111,
            nrows_ncols=(1, 3),
            axes_pad=(0.4, 0.1),
            label_mode="L",
            share_all=True,
            cbar_location="top",
            cbar_mode="each",
            cbar_size="5%",
            cbar_pad=0.3,
            )

errs = [xerr, yerr, rerr]
labs = ['X', 'Y', 'R']
lim = 1

vmax = np.abs(errs).max()
for i in range(3):
    vmin = [-vmax, 0][int(i==2)]
    out = grid[i].scatter(positions[:,0]*1e3, positions[:,1]*1e3, c=errs[i], \
        s=6, vmin=vmin, vmax=vmax,cmap=plt.cm.jet)
    grid[i].plot(0,0, 'r+')
    cbar = plt.colorbar(out, cax=grid[i].cax, orientation='horizontal', \
        label=f'{labs[i]} Error [m]')
    cbar.ax.xaxis.set_ticks_position('top')
    grid[i].grid()
    grid[i].set_xticks(np.linspace(-lim,lim,5))
    grid[i].set_yticks(np.linspace(-lim,lim,5))
    grid[i].set_xlim([-lim,lim])
    grid[i].set_ylim([-lim,lim])
    grid[i].set_xlabel('X Position [m]')

grid[0].set_ylabel('Y Position [m]')

#Histogram
figh, axesh = plt.subplots(1, figsize=(7,5))
bb = 20
axesh.hist(rerr, bins=bb, label='R err')
axesh.hist(xerr, bins=bb, alpha=0.5, label='X err')
axesh.hist(yerr, bins=bb, alpha=0.5, label='Y err')
axesh.set_xlabel('Error [m]')
axesh.set_ylabel('Occurences')
axesh.legend()

print(rerr.mean(), rerr.std())

if do_save:
    ext = f"{data_run.split('__')[1]}__{model}"
    fig.savefig(os.path.join(plot_dir, f'error_map__{ext}.png'), dpi=200)
    figh.savefig(os.path.join(plot_dir, f'histogram__{ext}.png'), dpi=200)
    
else:
    breakpoint()
