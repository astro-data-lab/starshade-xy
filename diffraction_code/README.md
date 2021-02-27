# Diffraction Calculations

This directory contains the code needed to generate simulated pupil images using
the Boundary Diffraction Wave code in `bdw.py`. The directory `xtras` contains the apodization profile of the lab starshade (`lab_ss.dat`), a pupil mask of the Roman Space Telescope secondary mirror (`pupil_mask.h5`), and a pre-calculated solution for testing (`test_data__lab_ss.h5`).

Getting started
---------------------
A sample script for running a `BDW` simulation is contained in `run_bdw.py`. The BDW
class contained in `bdw.py` accepts a dictionary of parameters for its instantiation;
default parameters are stored in `BDW`. An quick example for running a simulation is given by:

    from bdw import BDW

    #Specify simulation parameters
    params = {
        ### Lab ###
        'wave':             0.405e-6,       #Wavelength of light [m]
        'z0':               27.5,           #Source - starshade distance [m]
        'z1':               50.,            #Starshade - telescope distance [m]

        ### Telescope ###
        'tel_diameter':     5e-3,           #Telescope aperture diameter [m]
        'num_tel_pts':      64,             #Size of grid to calculate over pupil
        'tel_shift':        [0, 0],         #(x,y) shift of telescope to starshade-source [m]

        ### Starshade ###
        'apod_name':        'lab_ss',       #Apodization profile name. Options: ['lab_ss', 'circle']
        'with_spiders':     False,          #Superimpose secondary mirror spiders on pupil image?

        ### Saving ###
        'save_dir_base':    './',           #Base directory to save data
        'session':          '',             #Session name, i.e., subfolder to save data
        'save_ext':         '',             #Save extension to append to data
        'do_save':          True,           #Save data?
    }

    #Load BDW class
    bdw = BDW(params)

    #Run simulation
    bdw.run_sim()

To generate a new image at a different location, the telescope's position can be changed with `BDW.tel_shift` and a new image calculate through `img = BDW.calculate_diffraction()`. An example in quickly generating images is shown in `generage_images.py`. `BDW` uses minimal dependencies of `numpy` and `h5py` to save data; `scipy` is only used for the image of the secondary spiders.

Dependencies
--------------------
You will need:

- `numpy <http://www.numpy.org/>`
- `h5py <http://www.h5py.org>`

And optionally:
- `scipy <https://www.scipy.org>`
- `pytest <https://pypi.org/project/pytest/>`

Testing
---------------------
If _pytest_ is installed, testing can be done through:

    pytest test_bdw.py

Documentation
--------------
No public documentation yet :(

Contributors
------------
 - Primary author: Anthony Harness (Princeton University)
