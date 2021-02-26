"""
bdw.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 02-26-2021

Description: BDW uses the Boundary Diffraction Wave (BDW) algorithm of
    "E. Cady, Optics Express, 20, 14 (2012)" to calculate the Fresnel diffraction
    of an opaque occulter with a radial apodization profile. The apodization profile
    is specified by the parameter 'apod_name'. The input to the class is a dictionary
    of parameters (with defaults specified in BDW.set_parameters). The simulation
    is run with BDW.run_sim(). Saves: electric field at the pupil, x/y coordinates.
    WARNING: this version only works in the geometric shadow of a closed occulter shape.

"""

import numpy as np
import h5py
import os
import time
import multiprocessing
import itertools
from scipy.ndimage import affine_transform

class BDW(object):

    def __init__(self, params={}):
        self.set_parameters(params)
        self.setup_sim()

    ## Pupil coordinates ##

    @property
    def tel_pts_x(self):
        return self.tel_pts + self.shift[0]

    @property
    def tel_pts_y(self):
        return self.tel_pts + self.shift[1]

############################################
####    Initialization ####
############################################

    def set_parameters(self, params):
        #Default parameters
        def_pms = {
            ### Lab ###
            'wave':             0.405e-6,       #Wavelength of light [m]
            'z0':               27.5,           #Source - starshade distance [m]
            'z1':               50.,            #Starshade - telescope distance [m]
            ### Telescope ###
            'tel_diameter':     5e-3,           #Telescope aperture diameter [m]
            'num_tel_pts':      256,            #Size of grid to calculate over pupil
            'image_pad':        10,             #Padding in pupil image outside of aperture
            'shift':            [0,0],          #(x,y) shift of telescope relative to starshade-source line [m]
            'with_spiders':     False,          #Superimpose spiders on pupil image?
            ### Starshade ###
            'apod_name':        'lab_ss',       #Apodization profile name. Options: ['lab_ss', 'circle']
            'num_petals':       12,             #Number of starshade petals
            'circle_rad':       12.5e-3,        #Radius of circle occulter
            'num_occ_pts':      1000,           #Number of points in occulter (per petal edge if starshade)
            ### Saving ###
            'verbose':          True,           #Print out status statements?
            'save_dir_base':    './',           #Base directory to save data
            'session':          '',             #Session name, i.e., subfolder to save data
            'save_ext':         '',             #Save extension to append to data
            'do_save':          True,           #Save data?
            ### Misc ###
            'xtras_dir':        './xtras',      #Location of extras (apodization function, pupil mask)
            'allow_parallel':   True,           #Allow to be run in parallel?
        }

        #Set user and default parameters
        for k,v in {**def_pms, **params}.items():
            #Check if valid parameter name
            if k not in def_pms.keys():
                print(f'\nError: Invalid Parameter Name: {k}\n')
                import sys; sys.exit(0)

            setattr(self, k, v)

        #Create save directory
        if self.do_save:
            self.save_dir = f'{self.save_dir_base}/{self.session}'
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)

        #Wavenumber
        self.kk = 2.*np.pi / self.wave

        #Distance scalings
        self.zscl_a = self.z0 / (self.z0 + self.z1)
        self.zscl_b = self.z1 / (self.z0 + self.z1)
        self.zeff = self.z1 * self.zscl_a

        #Aperture points (include padding)
        self.image_width = self.tel_diameter * (1. + 2.*self.image_pad / self.num_tel_pts)
        self.num_pts = 2*self.image_pad + self.num_tel_pts
        self.tel_pts = self.image_width*(np.arange(self.num_pts)/self.num_pts - 0.5)

    def setup_sim(self):

        #Load occulter
        self.load_occulter()

        #Load pupil mask
        self.load_pupil_mask()

        ### Pre - Calculations ###

        #Exponential (QPF)
        self.exp_arg = 1j*self.kk/self.z1       #For cross terms
        self.pre_exp = 1j*self.kk/(2*self.zeff) * (self.loci**2).sum(1)

        #Top of inclination factor
        self.pre_top = \
            self.loci[:,1]*(self.dls[:,0]*(self.z0 + self.z1)) - \
            self.loci[:,0]*(self.dls[:,1]*(self.z0 + self.z1))

        #Bottom of inclination factor
        zfac = (self.z0**2 + self.z1**2.)/(2.*self.z0*self.z1)
        self.pre_bot = (self.loci**2).sum(1) * (1. + zfac)

############################################
############################################

############################################
####   Occulter ####
############################################

    def load_occulter(self):
        #Build circle or starshade
        if self.apod_name == 'circle':
            #Get midpoint edge locations and edge segment lengths
            self.loci, self.dls = self.build_circle()

        else:
            #Get midpoint edge locations and edge segment lengths
            self.loci, self.dls = self.build_starshade()

    def build_circle(self):
        #Build cartesian points
        the = np.linspace(0, 2*np.pi, self.num_occ_pts, endpoint=False)
        loci = self.circle_rad * np.stack((np.cos(the), np.sin(the)), 1)

        #Get midpoint scheme
        return self.get_midpoint_scheme(loci)

    def build_starshade(self):

        #Load occulter txt data (radius [um], apodization)
        rads, apod = np.genfromtxt(f'{self.xtras_dir}/{self.apod_name}.dat', delimiter=',').T

        #Convert to meters and angle
        rads *= 1e-6
        apod *= np.pi/self.num_petals

        #Resample onto smaller grid
        newr = np.linspace(rads.min(), rads.max(), self.num_occ_pts)
        newa = np.interp(newr, rads, apod)

        #Convert to cartesian coordinate
        xx = newr * np.cos(newa)
        yy = newr * np.sin(newa)

        #Flip over y to add other petal edge
        xx = np.concatenate((xx,  xx[::-1]))
        yy = np.concatenate((yy, -yy[::-1]))

        #Rotate and build each petal
        loci, dls = np.empty((0,2)), np.empty((0,2))
        for i in range(self.num_petals):
            #Rotate to new coordinates
            rot_ang = 2*np.pi/self.num_petals * i
            newx =  xx*np.cos(rot_ang) + yy*np.sin(rot_ang)
            newy = -xx*np.sin(rot_ang) + yy*np.cos(rot_ang)

            #Get midpoint scheme
            cur_loc, cur_dl = self.get_midpoint_scheme(np.stack((newx, newy), 1))

            #Append
            loci = np.concatenate((loci, cur_loc))
            dls = np.concatenate((dls, cur_dl))

        #Cleanup
        del rads, apod, newr, newa, newx, newy, xx, yy

        return loci, dls

    def get_midpoint_scheme(self, loci):
        #Rollover 1 point
        locr = np.concatenate((loci[1:], loci[:1]))

        #Get midpoint values
        mid_pt = (loci + locr)/2.

        #Calculate edge lengths
        dls = locr - loci

        return mid_pt, dls

############################################
############################################

############################################
####   Pupil Mask ####
############################################

    def load_pupil_mask(self):
        #Load Roman Space Telescope pupil
        if self.with_spiders:
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
####    Main script ####
############################################

    def run_sim(self):

        #Start
        start = time.perf_counter()
        if self.verbose:
            print(f"\nRunning '{self.apod_name}' on {self.num_pts} x {self.num_pts} grid...\n")

        #Calculate diffraction
        Emap = self.calculate_diffraction()

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
        #Get indices for x,y values
        inds = [(i,j) for i,j in itertools.product(range(self.num_pts), range(self.num_pts))]

        #Run in parallel
        if self.allow_parallel:

            #Get processor/chunk size
            procs = multiprocessing.cpu_count()
            chunksize = 2**8

            #Run in pool
            with multiprocessing.Pool(processes=procs) as pool:
                Emap = np.array(pool.map(self.calc_electric_field, inds, \
                    chunksize)).reshape((self.num_pts, self.num_pts))

        else:

            #Run serially
            Emap = np.fromiter(map(self.calc_electric_field, inds), \
                dtype=np.complex).reshape((self.num_pts, self.num_pts))

        #Add constants in front of integral
        Emap = self.add_leading_terms(Emap)

        #Add pupil mask
        Emap *= self.pupil_mask

        return Emap

    def calc_electric_field(self, xyind):
        #Get pupil coordinates (shifted)
        xx = self.tel_pts_x[xyind[1]]
        yy = self.tel_pts_y[xyind[0]]

        #Build exponential
        exp = np.exp(self.pre_exp - \
            self.loci[:,0]*(self.exp_arg*xx) - self.loci[:,1]*(self.exp_arg*yy))

        #Build top of inclination factor
        inc_top = self.pre_top + \
            (self.z0*xx)*self.dls[:,1] - (self.z0*yy)*self.dls[:,0]

        #Build bottom of inclination factor
        inc_bot = self.pre_bot + 0.5*self.z0/self.z1*(xx**2 + yy**2) + \
            - self.loci[:,0]*(xx*(1. + self.z0/self.z1)) \
            - self.loci[:,1]*(yy*(1. + self.z0/self.z1))

        #Build inclination factor
        inc = inc_top / inc_bot

        #Calculate integral with dot product
        ans = np.dot(exp, inc)

        return ans

    def add_leading_terms(self, Emap):
        #Add leading QPF phase
        ap_opd = (self.tel_pts_x**2 + self.tel_pts_y[:,None]**2) / self.z1
        Emap *= np.exp(1j*self.kk/2. * ap_opd)
        Emap *= np.exp(1j*self.kk * self.z1)

        #Divide by numerical constants (negative sign to match lotus)
        Emap /= -self.z1 * 4.*np.pi

        return Emap

############################################
############################################

############################################
####   Save Data ####
############################################

    def save_data(self, Emap):
        if not self.do_save:
            return

        #Extension
        save_ext = [self.save_ext, f'_{self.save_ext}'][int(self.save_ext != '')]

        #Save
        with h5py.File(f'{self.save_dir}/pupil{save_ext}.h5', 'w') as f:
            f.create_dataset('pupil_Ec', data = Emap)
            f.create_dataset('pupil_xx', data = self.tel_pts_x)
            f.create_dataset('pupil_yy', data = self.tel_pts_y)

############################################
############################################


if __name__ == '__main__':

    params = {
            'num_tel_pts':      64,
            'num_occ_pts':      1000,
            # 'apod_name':        'lab_ss',
            'apod_name':        'circle',
            'save_dir_base':    './Results',
            'do_save':          False,
            'with_spiders':     True,
    }

    bdw = BDW(params)
    emap = bdw.run_sim()

    import matplotlib.pyplot as plt;plt.ion()
    plt.imshow(np.abs(emap))
    breakpoint()
