"""
test_diffraq.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 06-11-2021

Description: Run tests of DIFFRAQ by comparing simulated results of a circular occulter
    to the anlytic solution and simulated results of the laboratory starshade
    to results pre-generated from a different package.

"""

import numpy as np
import diffraq
from scipy.special import jn
import h5py
import os
from diffraq import image_util

class Test_Diffraq(object):

    def run_all_tests(self):
        for tt in ['circle', 'starshade', 'focuser']:
            getattr(self, f'test_{tt}')()

############################################

    def test_circle(self):
        #Intensity tolerance
        itol = 1e-6

        #Diffraction parameters
        wave = 0.6e-6
        z1 = 50.
        z0 = 27.
        tel_diameter = 10e-3
        num_pts = 64
        circle_rad = 12e-3

        #Diffraq parameters
        params = {
                'wave':             wave,
                'z0':               z0,
                'z1':               z1,
                'tel_diameter':     tel_diameter,
                'num_tel_pts':      num_pts,
                'apod_name':        'circle',
                'circle_rad':       circle_rad,
                'image_pad':        0,
                'do_save':          False,
                'verbose':          False,
        }

        #Load Diffraq
        sim = diffraq.Simulator(params)

        #Loop over shifts
        for shift in [[0,0], [-3e-3, 0], [2e-3, -1.5e-3]]:

            #Loop over babinet
            for is_babinet in [False, True]:

                #Set babinet flag
                sim.is_babinet = is_babinet

                #Set sim's shift
                sim.tel_shift = shift

                #Run Diffraq simulation
                dsol = sim.run_sim()

                #Get analytic solution
                ss = np.hypot(sim.tel_pts_x, sim.tel_pts_y[:,None])
                asol = calculate_circle_solution(ss, wave, z1, z0, circle_rad, is_babinet)

                #Verify correct
                assert(np.abs(asol - dsol).max()**2 < itol)

############################################

    def test_starshade(self):
        #Tolerance (amplitude)
        tol = 1e-4

        #Apodization name
        apod_name = 'lab_ss'

        #Load test data
        fname = os.path.join(os.pardir, 'diffraction_code', 'xtras', \
            f'test_data__{apod_name}.h5')
        pms, imgs = {}, {}
        with h5py.File(fname, 'r') as f:
            #Get parameters
            for k in f.keys():
                #Save images separately
                if k.startswith('pupil'):
                    imgs[k] = f[k][()]
                else:
                    pms[k]= f[k][()]

        #Add extra pms
        pms['do_save'] = False
        pms['verbose'] = False
        pms['image_pad'] = 0
        waves = pms.pop('waves')
        pms.pop('num_occ_pts')

        #Loop over wavelengths and run sim
        for wave in waves:

            #Run Sim
            pms['wave'] = wave
            sim = diffraq.Simulator(pms)
            emap = sim.run_sim()

            #Compare to data
            cmap = imgs[f'pupil_{wave*1e9:.0f}']

            #Assert true
            assert(np.abs(emap - cmap).max() < tol)

############################################

    def test_focuser(self):
        waves = np.array([0.3e-6, 0.6e-6])

        #Gaussian illumination on pupil
        def gauss(num_pts, dx):
            xx = image_util.get_grid_points(num_pts, dx=dx)
            rr = np.hypot(xx, xx[:,None])
            pupil = np.exp(-rr**2/xx.max()**2) + 0j
            return pupil, xx

        #Loop through wavelengths
        for i in range(len(waves)):
            #Loop through magnifications
            for mag in [0.3, 2.]:

                #Build simulator
                sim = diffraq.Simulator({'wave':waves[i], 'pupil_mag':mag})

                #Build gaussian pupil image
                pupil, x0 = gauss(sim.num_pts, sim.focuser.dx0)

                #Get images
                image, grid_pts = sim.focuser.calculate_image(pupil)
                dx = grid_pts[1] - grid_pts[0]

                #Get new pupil (scaled gaussian)
                pup2, x2 = gauss(sim.num_pts, dx)
                pup2, dum = image_util.round_aperture(pup2)
                pup2 = abs(pup2)**2

                #Check sampling
                assert(np.isclose(dx, sim.focuser.dx0*mag))

                #Check if matches scaled gaussian
                assert(np.allclose(pup2, image))

        #Cleanup
        del pupil, image, pup2, x0, x2, grid_pts

############################################
##### Circle Analytical Functions #####
############################################

def calculate_circle_solution(ss, wave, zz, z0, circle_rad, is_opaque):
    """Calculate analytic solution to circular disk over observation points ss."""

    vu_brk = 0.99

    #Derived
    kk = 2.*np.pi/wave
    zeff = zz * z0 / (zz + z0)

    #Lommel variables
    uu = kk*circle_rad**2./zeff
    vv = kk*ss*circle_rad/zz

    #Get value of where to break geometric shadow
    vu_val = np.abs(vv/uu)

    #Build nominal map
    Emap = np.zeros_like(ss) + 0j

    #Calculate inner region (shadow for disk, illuminated for aperture)
    sv_inds = vu_val <= vu_brk
    Emap[sv_inds] = get_field(uu, vv, ss, sv_inds, kk, zz, z0, \
        is_opaque=is_opaque, is_shadow=is_opaque)

    #Calculate outer region (illuminated for disk, shadow for aperture)
    sv_inds = ~sv_inds
    Emap[sv_inds] = get_field(uu, vv, ss, sv_inds, kk, zz, z0, \
        is_opaque=is_opaque, is_shadow=not is_opaque)

    #Mask out circular aperture
    yind, xind = np.indices(Emap.shape)
    rhoi = np.hypot(xind - Emap.shape[0]/2, yind - Emap.shape[1]/2)
    Emap[rhoi >= min(Emap.shape[-2:])/2.] = 0.

    return Emap

def get_field(uu, vv, ss, sv_inds, kk, zz, z0, is_opaque=True, is_shadow=True):
    #Lommel terms
    n_lom = 50

    #Return empty if given empty
    if len(ss[sv_inds]) == 0:
        return np.array([])

    #Shadow or illumination? Disk or Aperture?
    if (is_shadow and is_opaque) or (not is_shadow and not is_opaque):
        AA, BB = lommels_V(uu, vv[sv_inds], nt=n_lom)
    else:
        BB, AA = lommels_U(uu, vv[sv_inds], nt=n_lom)

    #Flip sign for aperture
    if not is_opaque:
        AA *= -1.

    #Calculate field due to mask QPF phase term
    EE = np.exp(1j*uu/2.)*(AA + 1j*BB*[1.,-1.][int(is_shadow)])

    #Add illuminated beam
    if not is_shadow:
        EE += np.exp(-1j*vv[sv_inds]**2./(2.*uu))

    #Add final plane QPF phase terms
    EE *= np.exp(1j*kk*(ss[sv_inds]**2./(2.*zz) + zz))

    #Scale for diverging beam
    EE *= z0 / (zz + z0)

    return EE

def lommels_V(u,v,nt=10):
    VV_0 = 0.
    VV_1 = 0.
    for m in range(nt):
        VV_0 += (-1.)**m*(v/u)**(0+2.*m)*jn(0+2*m,v)
        VV_1 += (-1.)**m*(v/u)**(1+2.*m)*jn(1+2*m,v)
    return VV_0, VV_1

def lommels_U(u,v,nt=10):
    UU_1 = 0.
    UU_2 = 0.
    for m in range(nt):
        UU_1 += (-1.)**m*(u/v)**(1+2.*m)*jn(1+2*m,v)
        UU_2 += (-1.)**m*(u/v)**(2+2.*m)*jn(2+2*m,v)
    return UU_1, UU_2

############################################
############################################


if __name__ == '__main__':

    tt = Test_Diffraq()
    tt.run_all_tests()
