import numpy as np
import matplotlib.pyplot as plt
import h5py

data_run = 'run__6_01_21__data_1s_bin1__spiders__median'

do_save = [False, True][0]

with h5py.File(f'./Results/{data_run}.h5', 'r') as f:
    xerr = f['xerr'][()]
    yerr = f['yerr'][()]
    positions = f['positions'][()]

dist = np.hypot(xerr, yerr)

if not do_save:
    plt.ion()

# plt.scatter(xerr, yerr, marker = '+')
# plt.grid()
# plt.title('Scatterplot of Errors (SNR = 3.5)')
# plt.xlabel('Error in x (m)')
# plt.ylabel('Error in y (m)')

fig, axes = plt.subplots(1, figsize=(6,6))
cbar = plt.colorbar(axes.scatter(positions[:,0]*1e3, positions[:,1]*1e3, c=dist, s=2))
cbar.set_label('Distance Error [m]')
axes.grid()
axes.set_xlabel('X Position [m]')
axes.set_ylabel('Y Position [m]')

fig2, axes2 = plt.subplots(1, figsize=(6,6))
bb = 20
axes2.hist(dist, bins=bb, label='R err')
axes2.hist(xerr, bins=bb, alpha=0.5, label='X err')
axes2.hist(yerr, bins=bb, alpha=0.5, label='Y err')
axes2.legend()

if do_save:
    fig.savefig(f'./Plots/pos_v_err__{data_run}.png', dpi=200)
    fig2.savefig(f'./Plots/hist__{data_run}.png', dpi=200)
else:
    breakpoint()
