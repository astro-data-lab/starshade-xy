import numpy as np
import h5py
import time
import os
import imp
imp_dir = '/home/aharness/repos/starshade-xy/quadrature_code'
diffraq = imp.load_source("diffraq", os.path.join(imp_dir, 'diffraq', "__init__.py"))
new_noise_maker = imp.load_source("new_noise_maker", os.path.join(imp_dir, "new_noise_maker.py"))
from new_noise_maker import Noise_Maker

#Save directory
session = 'n10k_sigd25'

#Number of samples
num_imgs = 10000

#Gaussian distribution
sig_width = 0.25

#SNR range
snr_range = [5, 25]

#User options
apod_name = 'm12p8'
with_spiders = True
wave = 403e-9
num_tel_pts = 96
#Telescope sizes in Lab and Space coordinates [m] (sets scaling factor)
Dtel_lab = 2.201472e-3
Dtel_space = 2.4
#Random number generator seed
seed = 88

############################

#Get random number generatore
rng = np.random.default_rng(seed)

#Lab to space scaling
lab2space = Dtel_space / Dtel_lab
space2lab = 1/lab2space

############################

#Create directory
save_dir = './Sim_Data'

#Load instance of noise maker
noise_params = {'count_rate': 7, 'rng': rng, 'snr_range': snr_range,
    'num_tel_pts': num_tel_pts}
noiser = Noise_Maker(noise_params)

#Specify simulation parameters
params = {
    ### Lab ###
    'wave':             wave,                   #Wavelength of light [m]

    ### Telescope ###
    'tel_diameter':     Dtel_lab,               #Telescope aperture diameter [m]
    'num_tel_pts':      num_tel_pts,            #Size of grid to calculate over pupil

    ### Starshade ###
    #will specify apod_name after circle is run
    'num_petals':       12,                     #Number of starshade petals

    ### Saving ###
    'do_save':          False,                  #Don't save data
    'verbose':          False,                  #Silence output
    'xtras_dir':        f'{imp_dir}/xtras',
}

#Run unblocked image first (lab calibration)
params['apod_name'] = 'circle'
params['circle_rad'] = 25.086e-3
sim = diffraq.Simulator(params)
sim.setup_sim()
cal_img = np.abs(sim.calculate_diffraction())**2

#Set to noise maker
noiser.set_suppression_norm(cal_img)

#Save image
cal_file = os.path.join(save_dir, 'calibration')

#New simulator for starshade images
params['with_spiders'] = with_spiders
params['apod_name'] = apod_name
sim = diffraq.Simulator(params)
sim.setup_sim()

#Initiate containers
images = np.empty((num_imgs, sim.num_pts, sim.num_pts))
positions = np.empty((num_imgs, 2))

tik = time.perf_counter()

#Loop and calculate image
for i in range(num_imgs):

    if i % (num_imgs // 10) == 0:
        print(f'Running step # {i} / {num_imgs} ({time.perf_counter()-tik:.0f} s)')

    #Get random position from gaussian
    nx, ny = rng.multivariate_normal([0,0], np.eye(2)*(sig_width*space2lab)**2)

    #Set shift of telescope
    sim.tel_shift = [nx, ny]

    #Get random snr
    snr = rng.uniform(snr_range[0], snr_range[1])

    #Get diffraction and convert to intensity
    img = np.abs(sim.calculate_diffraction())**2

    #Add noise
    img = noiser.add_noise(img, snr)

    #Store
    images[i] = img
    positions[i] = [nx,ny]

tok = time.perf_counter()

#Save
with h5py.File(f'{save_dir}/{session}_data.h5', 'w') as f:
    f.create_dataset('num_tel_pts', data=num_tel_pts)
    f.create_dataset('sig_width', data=sig_width)
    f.create_dataset('lab2space', data=lab2space)
    f.create_dataset('seed', data=seed)
    f.create_dataset('snr', data=snr)
    f.create_dataset('images', data=images)
    f.create_dataset('positions', data=positions)

print(f'\nRan {num_imgs} images in {tok-tik:.0f} s\n')
