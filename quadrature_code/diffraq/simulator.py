"""
simulator.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 06-11-2021

Description: DIFFRAQ uses areal quadrature and non-uniform FFTs to efficiently
    calculate Fresnel diffraction. Based off the code FRESNAQ (Barnett 2021):
    https://github.com/ahbarnett/fresnaq

"""

import numpy as np
import h5py
import time
import os
import diffraq
from scipy.interpolate import InterpolatedUnivariateSpline

class Simulator(object):

    def __init__(self, params={}):
        self.set_parameters(params)

############################################
####    Initialization ####
############################################

    def set_parameters(self, params):
        #Default parameters
        def_pms = {
            ### Lab ###
            'wave':             0.405e-6,       #Wavelength of light [m]
            'z0':               27.455,         #Source - starshade distance [m]
            'z1':               50.,            #Starshade - telescope distance [m]
            ### Telescope ###
            'tel_diameter':     5e-3,           #Telescope aperture diameter [m]
            'num_tel_pts':      256,            #Size of grid to calculate over pupil
            'image_pad':        10,             #Padding in pupil image outside of aperture
            'tel_shift':        [0,0],          #(x,y) shift of telescope relative to starshade-source line [m]
            'with_spiders':     False,          #Superimpose secondary mirror spiders on pupil image?
            'skip_mask':        False,          #Skip pupil mask entirely
            ### Starshade ###
            'apod_name':        'lab_ss',       #Apodization profile name. Options: ['lab_ss', 'circle']
            'num_petals':       12,             #Number of starshade petals
            'circle_rad':       12.5e-3,        #Radius of circle occulter
            'radial_nodes':     500,            #Number of radial quadrature points
            'theta_nodes':      50,             #Number of theta quadrature points
            'fft_tol':          1e-9,           #Tolerance on finufft
            'is_babinet':       False,          #Use Babinet's principle?
            ### Saving ###
            'verbose':          True,           #Print out status statements?
            'save_dir_base':    './',           #Base directory to save data
            'session':          '',             #Session name, i.e., subfolder to save data
            'save_ext':         '',             #Save extension to append to data
            'do_save':          True,           #Save data?
            ### Misc ###
            'xtras_dir':        './xtras',      #Location of extras (apodization function, pupil mask)
        }

        #Set user and default parameters
        for k,v in {**def_pms, **params}.items():
            #Check if valid parameter name
            if k not in def_pms.keys():
                print(f'\nError: Invalid Parameter Name: {k}\n')
                import sys; sys.exit(0)

            #Set parameter value as attribute
            setattr(self, k, v)

        #Create save directory
        if self.do_save:
            self.save_dir = f'{self.save_dir_base}/{self.session}'
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        #Diverging beam
        self.z_scl = self.z0 / (self.z0 + self.z1)

        #Add image pad
        self.num_pts = 2*self.image_pad + self.num_tel_pts

        #Aperture points
        self.tel_pts = self.get_grid_points(self.num_pts, width=self.tel_diameter)

    def setup_sim(self):
        #Load pupil mask
        self.load_pupil_mask()

        #Build quadrature
        self.build_quadrature()

############################################
############################################

############################################
####    Main script ####
############################################

    def run_sim(self):
        #Start
        start = time.perf_counter()
        if self.verbose:
            print(f"\nRunning '{self.apod_name}' on {self.num_pts} x {self.num_pts} grid...\n")

        #Setup
        self.setup_sim()

        #Calculate diffraction
        Emap = self.calculate_diffraction()

        #Store position
        self.tel_pts_x, self.tel_pts_y = self.tel_pts + np.array(self.tel_shift)[:,None]

        #Print out finish time
        if self.verbose:
            print(f'\nDone! Time: {time.perf_counter() - start:.2f} [s]\n')

        #Save data
        self.save_data(Emap)

        return Emap

############################################
############################################

############################################
####   Diffraction Calcs ####
############################################

    def calculate_diffraction(self):

        #Adjust occulter values if off_axis (shift doesn't go into beam function)
        if not np.isclose(0, np.hypot(*self.tel_shift)):
            xq = self.xq - self.tel_shift[0] * self.z_scl
            yq = self.yq - self.tel_shift[1] * self.z_scl
            xoff = 2*(self.tel_pts*self.tel_shift[0] + self.tel_pts[:,None]*self.tel_shift[1])
            xoff += np.hypot(*self.tel_shift)**2
        else:
            xq = self.xq
            yq = self.yq
            xoff = 0

        #lambda * z
        lamzz = self.wave * self.z1
        lamz0 = self.wave * self.z0

        #Calculate diffraction
        uu = diffraq.diffract_grid(xq, yq, self.wq, lamzz, \
            self.tel_pts, self.fft_tol, lamz0=lamz0, is_babinet=self.is_babinet)

        #Account for extra phase added by off_axis
        uu *= np.exp(1j*np.pi/lamz0*self.z_scl * xoff)

        #Multiply by plane wave
        uu *= np.exp(1j * 2*np.pi/self.wave * self.z1)

        #Add pupil mask
        uu *= self.pupil_mask

        return uu

############################################
############################################

############################################
####   Quadrature ####
############################################

    def build_quadrature(self):
        if self.apod_name == 'circle':
            self.build_circle_quadrature()
        else:
            self.build_starshade_quadrature()

    def build_starshade_quadrature(self):

        #Load apod data
        fname = f'{self.xtras_dir}/apod__{self.apod_name}.h5'
        with h5py.File(fname, 'r') as f:
            data = f['data'][()]
            self.has_center = f['has_center'][()]
            self.is_babinet = f['is_babinet'][()]

        #Set min/max radius
        self.min_radius = data[:,0].min()
        self.max_radius = data[:,0].max()

        #Interpolate data
        self.apod_func = InterpolatedUnivariateSpline(data[:,0], data[:,1], k=2, ext=3)

        #Cleanup
        del data

        #Get quadrature points
        self.xq, self.yq, self.wq = diffraq.petal_quad(self.apod_func, self.num_petals, \
            self.min_radius, self.max_radius, self.radial_nodes, self.theta_nodes, \
            has_center=self.has_center)

    def build_circle_quadrature(self):
        func = lambda t: self.circle_rad * np.ones_like(t)
        self.xq, self.yq, self.wq = diffraq.polar_quad(func, 50, 50)

############################################
############################################

############################################
####   Pupil Mask ####
############################################

    def load_pupil_mask(self):
        #Load Roman Space Telescope pupil
        if self.with_spiders and has_scipy:
            #Load Pupil Mask
            with h5py.File(f'{self.xtras_dir}/pupil_mask.h5', 'r') as f:
                full_mask = f['mask'][()]

            #Do affine transform
            scaling = full_mask.shape[0]/self.num_tel_pts
            dx = -self.image_pad*scaling
            affmat = np.array([[scaling, 0, dx], [0, scaling, dx]])
            self.pupil_mask = affine_transform(full_mask, affmat, \
                output_shape=(self.num_pts, self.num_pts), order=5)

            #Mask out
            self.pupil_mask[self.pupil_mask < 0] = 0
            self.pupil_mask[self.pupil_mask > 1] = 1

        elif self.skip_mask:
            #Don't have any mask
            self.pupil_mask = np.ones((self.num_pts, self.num_pts))

        else:
            #Build round aperture for mask
            self.pupil_mask = np.ones((self.num_pts, self.num_pts))

            #Build radius values
            rhoi = np.hypot(self.tel_pts, self.tel_pts[:,None])

            #Block out circular aperture
            self.pupil_mask[rhoi >= self.tel_diameter/2] = 0.

            #Cleanup
            del rhoi

############################################
############################################

############################################
####    Extras ####
############################################

    def get_grid_points(self, ngrid, width=None, dx=None):
        #Handle case for width supplied
        if width is not None:
            grid_pts = width*(np.arange(ngrid)/ngrid - 0.5)
            dx = width/ngrid

        #Handle case for spacing supplied
        elif dx is not None:
            grid_pts = dx*(np.arange(ngrid) - 0.5*ngrid)

        #Handle the odd case
        if ngrid % 2 == 1:
            #Shift for odd points
            grid_pts += dx/2

        return grid_pts

    def save_data(self, Emap):
        if not self.do_save:
            return

        #Extension
        save_ext = [self.save_ext, f'_{self.save_ext}'][int(self.save_ext != '')]

        #Save
        with h5py.File(f'{self.save_dir}/diffraq_pupil{save_ext}.h5', 'w') as f:
            f.create_dataset('pupil_Ec', data = Emap)
            f.create_dataset('pupil_x', data = self.tel_pts_x)
            f.create_dataset('pupil_y', data = self.tel_pts_y)

############################################
############################################
