import numpy as np
import os
import imp

#Import diffraq
diff_dir = os.path.join(os.pardir, os.pardir, os.pardir, "quadrature_code")
diffraq = imp.load_source("diffraq", os.path.join(diff_dir,"diffraq","__init__.py"))

#Save directory
base_dir = './Sml_Data'

base_name = 'sim_check'

print(f'\nRunning {base_name} ...')

#Create directory
save_dir = f'{base_dir}/{base_name}'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)

#Number of steps
nsteps = 50

#Width of motion grid [m]
width = 1.5e-3

#Spacing between position steps [m]
dstep = width / nsteps

#Radius of random perturbations [m]
rad = np.sqrt(2) / 2 * dstep

tel_diam = 5e-3
dx = 2.2932e-05
num_tel_pts = int(tel_diam/dx)
image_pad = (250 - num_tel_pts)//2

#Specify simulation parameters
params = {
    ### Lab ###
    'wave':             0.403e-6,       #Wavelength of light [m]

    ### Telescope ###
    'tel_diameter':     tel_diam,           #Telescope aperture diameter [m]
    'num_tel_pts':      num_tel_pts,            #Size of grid to calculate over pupil
    'with_spiders':     False,          #Superimpose spiders on pupil image?
    'image_pad':        image_pad,

    ### Starshade ###
    'apod_name':        'm12p8',       #Apodization profile name. Options: ['lab_ss', 'circle']
    'num_petals':       12,             #Number of starshade petals

    ### Saving ###
    'do_save':          False,          #Don't save data
    'verbose':          False,          #Silence output
    'xtras_dir':        os.path.join(diff_dir,"xtras"),

}

#Load simulator
sim = diffraq.Simulator(params)
sim.setup_sim()

#Build steps
steps = np.linspace(-width/2, width/2, num=nsteps)

#Containers
images = np.empty((0, sim.num_pts, sim.num_pts))
positions = np.empty((0, 2))

#Create new csv file
csv_file = f'{save_dir}/{base_name}.csv'
with open(csv_file, 'w') as f:
    pass

#Loop over steps in each axis and calculate image
i = 0
for x in steps:
    print(f'Running x step # {i // len(steps) + 1} / {len(steps)}')
    for y in steps:

        #Set shift of telescope
        nx = x + 2 * rad * (np.random.random_sample() - 0.5)
        ny = y + 2 * rad * (np.random.random_sample() - 0.5)
        sim.tel_shift = [nx, ny]

        #Get diffraction and convert to intensity
        img = np.abs(sim.calculate_diffraction())**2

        #Number string
        num_str = str(i).zfill(6)

        #Save and write position to csv
        np.save(f'{save_dir}/{num_str}', img)
        with open(csv_file, 'a') as f:
            f.write(f'{num_str}, {nx}, {ny}\n')
        i += 1
