import numpy as np
import matplotlib.pyplot as plt
import h5py

data_run = 'run__6_01_21__data_1s_bin1__spiders__median'

model = 'newest_2'
# model = 'wide_1d5'
model = 'big'
# model = 'dx_shift'

load_ext = ['', '_spid'][0]

do_save = [False, True][0]

with h5py.File(f'./Results/{data_run}__{model}{load_ext}.h5', 'r') as f:
    xerr = f['xerr'][()]
    yerr = f['yerr'][()]
    positions = f['positions'][()]

rerr = np.hypot(xerr, yerr)

if not do_save:
    plt.ion()

fig, axes = plt.subplots(1, figsize=(6,6))
cbar = plt.colorbar(axes.scatter(positions[:,0]*1e3, positions[:,1]*1e3, c=rerr, s=6))
cbar.set_label('R Error [m]')
axes.grid()
axes.set_xlabel('X Position [m]')
axes.set_ylabel('Y Position [m]')
axes.plot(0,0, 'r+')
lim = 1
axes.set_xlim([-lim,lim])
axes.set_ylim([-lim,lim])

if [False, True][1]:
    figx, axesx = plt.subplots(1, figsize=(6,6))
    cbarx = plt.colorbar(axesx.scatter(positions[:,0]*1e3, positions[:,1]*1e3, c=xerr, s=6))
    cbarx.set_label('X Error [m]')
    axesx.grid()
    axesx.set_xlabel('X Position [m]')
    axesx.set_ylabel('Y Position [m]')
    axesx.plot(0,0, 'r+')
    lim = 1
    axesx.set_xlim([-lim,lim])
    axesx.set_ylim([-lim,lim])

    figy, axesy = plt.subplots(1, figsize=(6,6))
    cbary = plt.colorbar(axesy.scatter(positions[:,0]*1e3, positions[:,1]*1e3, c=yerr, s=6))
    cbary.set_label('Y Error [m]')
    axesy.grid()
    axesy.set_xlabel('X Position [m]')
    axesy.set_ylabel('Y Position [m]')
    axesy.plot(0,0, 'r+')
    lim = 1
    axesy.set_xlim([-lim,lim])
    axesy.set_ylim([-lim,lim])

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
    fig.savefig(f'./Plots/pos_v_err__{data_run}.png', dpi=200)
    fig2.savefig(f'./Plots/hist__{data_run}.png', dpi=200)
else:
    breakpoint()
