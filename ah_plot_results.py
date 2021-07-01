import numpy as np
import matplotlib.pyplot as plt
import h5py
from mpl_toolkits.axes_grid1 import ImageGrid

data_run = 'run__6_01_21__data_1s_bin1__spiders__median'

model = 'newest_2'
model = 'wide_1d5'
# model = 'wide'
# model = 'big'
model = 'dx_shift'

# data_run = 'run__6_01_21__data_1s_bin1__round__median'
# model = 'round_dx'

load_ext = ['', '_spid'][0]

do_save = [False, True][0]

with h5py.File(f'./Results/{data_run}__{model}{load_ext}.h5', 'r') as f:
    xerr = f['xerr'][()]
    yerr = f['yerr'][()]
    positions = f['positions'][()]

rerr = np.hypot(xerr, yerr)

if not do_save:
    plt.ion()

fig = plt.figure(1, figsize=(11,7))
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
    if i == 2:
        vmin = 0
    else:
        vmin = -vmax
    out = grid[i].scatter(positions[:,0]*1e3, positions[:,1]*1e3, c=errs[i], \
        s=6, vmin=vmin, vmax=vmax)
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

# figr, axesr = plt.subplots(1, figsize=(6,6))
# cbarr = plt.colorbar(axesr.scatter(positions[:,0]*1e3, positions[:,1]*1e3, c=rerr, s=6))
# cbarr.set_label('R Error [m]')
# axesr.grid()
# axesr.set_xlabel('X Position [m]')
# axesr.set_ylabel('Y Position [m]')
# axesr.plot(0,0, 'r+')
# lim = 1
# axesr.set_xlim([-lim,lim])
# axesr.set_ylim([-lim,lim])
#
# figx, axesx = plt.subplots(1, figsize=(6,6))
# cbarx = plt.colorbar(axesx.scatter(positions[:,0]*1e3, positions[:,1]*1e3, c=xerr, s=6))
# cbarx.set_label('X Error [m]')
# axesx.grid()
# axesx.set_xlabel('X Position [m]')
# axesx.set_ylabel('Y Position [m]')
# axesx.plot(0,0, 'r+')
# lim = 1
# axesx.set_xlim([-lim,lim])
# axesx.set_ylim([-lim,lim])
#
# figy, axesy = plt.subplots(1, figsize=(6,6))
# cbary = plt.colorbar(axesy.scatter(positions[:,0]*1e3, positions[:,1]*1e3, c=yerr, s=6))
# cbary.set_label('Y Error [m]')
# axesy.grid()
# axesy.set_xlabel('X Position [m]')
# axesy.set_ylabel('Y Position [m]')
# axesy.plot(0,0, 'r+')
# lim = 1
# axesy.set_xlim([-lim,lim])
# axesy.set_ylim([-lim,lim])

fig2, axes2 = plt.subplots(1, figsize=(6,6))
bb = 20
axes2.hist(rerr, bins=bb, label='R err')
axes2.hist(xerr, bins=bb, alpha=0.5, label='X err')
axes2.hist(yerr, bins=bb, alpha=0.5, label='Y err')
axes2.set_xlabel('Error [m]')
axes2.set_ylabel('Occurences')
axes2.legend()

print(rerr.mean(), rerr.std())

if do_save:
    ext = f"{data_run.split('__')[1]}__{model}"
    fig.savefig(f'./Plots/error_map__{ext}.png', dpi=200)
    # figr.savefig(f'./Plots/R_err__{ext}.png', dpi=200)
    # figx.savefig(f'./Plots/X_err__{ext}.png', dpi=200)
    # figy.savefig(f'./Plots/Y_err__{ext}.png', dpi=200)
    fig2.savefig(f'./Plots/hist__{ext}.png', dpi=200)
else:
    breakpoint()
