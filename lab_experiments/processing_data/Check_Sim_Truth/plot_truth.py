import numpy as np
import matplotlib.pyplot as plt
import h5py
from mpl_toolkits.axes_grid1 import ImageGrid

run = 'Sml_Noisy_Data'

do_save = [False, True][0]

if not do_save:
    plt.ion()

with h5py.File(f'./Truth_Results/truth__{run}.h5', 'r') as f:
    sim_pos = f['sim_pos'][()] * 1e3
    sensor = f['sensor'][()] * 1e3

#Calc errors
xerr = (sensor - sim_pos)[:,0]
yerr = (sensor - sim_pos)[:,1]
rerr = np.hypot(xerr, yerr)

print(rerr.mean(), rerr.std())

figh, axes = plt.subplots(1, figsize=(7,5))
axes.hist(rerr, bins=20, label='R err')
axes.hist(xerr, bins=20, alpha=0.5, label='X err')
axes.hist(yerr, bins=20, alpha=0.5, label='Y err')
axes.set_xlabel('Error [m]')
axes.set_ylabel('Occurences')
axes.legend()

figm = plt.figure(2, figsize=(11,5))
grid = ImageGrid(figm, 111,
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
    out = grid[i].scatter(sim_pos[:,0], sim_pos[:,1], c=errs[i], s=6, \
        vmin=vmin, vmax=vmax, cmap=plt.cm.jet)
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

if do_save:
    figm.savefig(f'./Plots/sim_truth_map.png', dpi=200)
    figh.savefig(f'./Plots/sim_truth_hist.png', dpi=200)
else:
    breakpoint()
