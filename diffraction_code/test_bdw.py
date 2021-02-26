import numpy as np
from bdw import BDW
from scipy.special import jn

class Test_BDW(object):

    def test_bdw(self):
        #Intensity tolerance
        itol = 1e-6

        #Diffraction parameters
        wave = 0.6e-6
        z1 = 50.
        z0 = 27.
        tel_diameter = 10e-3
        num_pts = 64
        circle_rad = 12e-3

        #BDW parameters
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

        #Load BDW
        bdw = BDW(params)

        #Loop over shifts
        for shift in [[0,0], [-3e-3, 0], [0,2e-3], [2e-3, -1.25e-3]]:

            #Set BDW's shift
            bdw.shift = shift

            #Run BDW simulation
            bsol = bdw.run_sim()

            #Get analytic solution
            ss = np.hypot(bdw.tel_pts_x, bdw.tel_pts_y[:,None])
            asol = calculate_circle_solution(ss, wave, z1, z0, circle_rad, True)

            #Verify correct
            assert(np.abs(asol - bsol).max()**2 < itol)

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

    tt = Test_BDW()
    tt.test_bdw()
