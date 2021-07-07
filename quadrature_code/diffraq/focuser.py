"""
focuser.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 01-18-2021
Package: DIFFRAQ
License: Refer to $pkg_home_dir/LICENSE

Description: Class to propagate the diffracted field to the focal plane of the
    target imaging system.
"""

import numpy as np
from diffraq import image_util
import scipy.fft as fft

class Focuser(object):

    def __init__(self, sim):
        self.sim = sim
        self.set_derived_parameters()

############################################
#####  Setup #####
############################################

    def set_derived_parameters(self):
        #Copy over from sim
        cp_pms = ['focal_length', 'pupil_mag', 'num_pts']
        for k in cp_pms:
            setattr(self, k, getattr(self.sim, k))

        #Hardwired focus at source
        object_distance = self.sim.z0 + self.sim.z1
        self.image_distance = 1./(1./self.focal_length - 1./object_distance)

        #Secondary lens
        self.focal_length_2nd = self.image_distance * \
            self.pupil_mag/(1 + self.pupil_mag)**2

        #Input Spacing
        self.dx0 = self.sim.tel_diameter / self.sim.num_tel_pts

############################################
############################################

############################################
####	Main Function ####
############################################

    def calculate_image(self, in_pupil):
        #Create copy
        pupil = in_pupil.copy()
        del in_pupil

        #Round aperture and get number of points
        NN = self.num_pts
        NN0 = pupil.shape[-1]
        pupil, dummy = image_util.round_aperture(pupil)

        #Create input plane indices
        et = (np.arange(NN0)/NN0 - 0.5) * NN0

        #Create focal plane indices
        xx = (np.arange(NN)/NN - 0.5) * NN

        #Propagate to first image plane
        img, dx = self.propagate_lens_diffraction(pupil, \
            self.sim.wave, et, xx, NN, self.dx0, is_2nd=False)

        #Apply wavefront error
        if self.sim.wfe_modes is not None:
            img *= self.get_wfe(xx, self.sim.wave)

        #Propagate to second image plane
        img, dx = self.propagate_lens_diffraction(img, \
            self.sim.wave, et, xx, NN, dx, is_2nd=True)

        #Turn into intensity
        img = np.real(img.conj()*img)

        #Use mean sampling to calculate grid pts (this grid matches fft.fftshift(fft.fftfreq(NN, d=1/NN)))
        grid_pts = image_util.get_grid_points(img.shape[-1], dx=dx)

        #Cleanup
        del et, pupil, xx

        return img, grid_pts

############################################
############################################

############################################
####	Image Propagation ####
############################################

    def propagate_lens_diffraction(self, pupil, wave, et, xx, NN, dx0, is_2nd=False):

        #Get focal length and propagation distance
        if is_2nd:
            ff = self.focal_length_2nd
            zz = self.focal_length_2nd * (1. + self.pupil_mag)
            normalization = 1
        else:
            ff = self.focal_length
            zz = self.focal_length_2nd * (1. + 1/self.pupil_mag)
            normalization = NN**2

        #Get output plane sampling
        dx = wave*zz/(dx0*NN)

        #Multiply by propagation kernels (lens and Fresnel)
        pupil *= self.propagation_kernel(et, dx0, wave, -ff)
        pupil *= self.propagation_kernel(et, dx0, wave,  zz)

        #Run FFT
        FF = self.do_FFT(pupil)

        #Multiply by Fresnel diffraction phase postfactor
        FF *= self.propagation_kernel(xx, dx, wave, zz)

        #Multiply by constant phase term
        FF *= np.exp(1j * 2.*np.pi/wave * zz)

        #Normalize such that peak is 1. Needs to scale for wavelength relative to other images
        FF /= normalization

        return FF, dx

############################################
############################################

############################################
####	Misc Functions ####
############################################

    def propagation_kernel(self, et, dx0, wave, distance):
        return np.exp(1j * 2.*np.pi/wave * dx0**2. * (et[:,None]**2 + et**2) / (2.*distance))

    def do_FFT(self, MM):
        return fft.ifftshift(fft.fft2(fft.fftshift(MM), workers=-1))

############################################
############################################

############################################
####	Wavefront Error ####
############################################

    def get_wfe(self, xx, wave):
        #Build radii, angle
        rad = np.hypot(xx, xx[:,None])
        rad /= rad.max()
        phi = np.arctan2(xx[:,None], xx)
        phase = 0
        for zm, zn, amp in self.sim.wfe_modes:
            phase += 2.*np.pi/wave * self.zernike_poly(rad, phi, zm, zn) * amp
        return np.exp(1j*phase)

    def zernike_rad(self, rho, m, n):
        coeff = 0.
        if (n - m) % 2 == 0:
            for k in range((n-m)//2+1):
                coeff += (-1)**k * np.math.comb(n-k, k) * \
                    np.math.comb(n-2*k, (n-m)//2 - k) * rho**(n-2*k)
        coeff[rho == 1] = 1
        return coeff

    def zernike_poly(self, rho, phi, m, n):
        if n < 0:
            return 0.
        if m >= 0:
            zz = self.zernike_rad(rho, np.abs(m), n) * np.cos(np.abs(m)*phi)
        else:
            zz = self.zernike_rad(rho, np.abs(m), n) * np.sin(np.abs(m)*phi)
        return zz

############################################
############################################
