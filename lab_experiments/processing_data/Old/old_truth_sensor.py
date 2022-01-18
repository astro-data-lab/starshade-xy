"""
truth_sensor.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 05-04-2021

Description: Class that holds functions to extract the position from a pupil image
    via other means such as centroiding or non-linear least squares fit to a model.

"""

import numpy as np
import h5py
from scipy.special import j0, j1
from scipy.optimize import least_squares
import time
import matplotlib.pyplot as plt;plt.ion()

class Truth_Sensor(object):

    def __init__(self, params={}):
        self.set_parameters(params)

    def set_parameters(self, params):
        def_pms = {
            ### Loading / Saving ###
            'image_file':       '',
            'debug':            False,
            'do_save':          False,
            'save_dir':         './Truth_Results',
            ### Sensing ###
            'sensing_method':   'model',        #Options: ['model', 'centroid']
            'cen_threshold':    0.75,           #Centroiding threshold
            ### Laboratory ###
            'wave':             405e-9,
            'ss_radius':        10.1e-3*np.sqrt(680/638),
            'z0':               27.455,
            'z1':               50.,
            'mask':             'none',         #Add spider masks in future
            'ccd_dark':         7e-4,           #Dark noise [e/px/s]
            'ccd_read':         4.78,           #Read noise [e/px/frame]
            'ccd_cic':          0.0025,         #CIC noise [e/px/frame]
            'ccd_gain':         0.768,          #inverse gain [ct/e]
        }

        #Set default parameters
        for k, v in def_pms.items():
            setattr(self, k, v)

        #Set user-specified parameters (if valid name)
        for k, v in params.items():
            if k not in def_pms.keys():
                print(f'\n!!! Bad parameter name: {k} !!!\n')
                breakpoint()
            else:
                setattr(self, k, v)
                #Update defaults
                def_pms[k] = v

        #Store
        self.params = def_pms

############################################
#####  Setup #####
############################################

    def setup(self):
        #Load data
        keys = ['num_tel_pts', 'image_pad', 'tel_diameter', 'positions', \
            'base_num_pts', 'binning', 'meta']
        with h5py.File(self.image_file, 'r') as f:
            for k in keys:
                setattr(self, k, f[k][()])
            self.img_shp = f['images'].shape[-2:]

        #Pupil magnification [m/pixel]
        self.pupil_mag = self.tel_diameter / self.base_num_pts

        #Get position arrays
        cen = np.array(self.img_shp)/2
        self.yy, self.xx = (np.indices(self.img_shp).T + cen[::-1] - \
            np.array(self.img_shp)).T

        #Flip to match image
        self.yy = self.yy[::-1,::-1]
        self.xx = self.xx[::-1,::-1]

        #Flatten and add binning
        self.xx = self.xx.flatten() * self.binning
        self.yy = self.yy.flatten() * self.binning

        #wavenumber * radius * z [in pixel units]
        self.kRz = 2.*np.pi/self.wave * self.ss_radius * self.pupil_mag / self.z1

    def load_image(self, i):
        with h5py.File(self.image_file, 'r') as f:
            img = f['images'][i]
        return img

############################################
############################################

############################################
####	Main Script ####
############################################

    def get_positions(self):

        print('\nGetting Truth Values...')
        tik = time.perf_counter()

        #Run setup
        self.setup()

        #Loop through images and calculate true positions
        truth = np.empty((0,2))
        errs = np.empty((0,2))
        flags = np.array([])
        for i in range(len(self.meta)):

            #Get current image
            img = self.load_image(i)

            #Get image error
            det_var = self.ccd_dark*self.meta[i][0] + self.ccd_cic**2. + \
                self.ccd_read**2.

            #Get initial guess
            pos0 = self.positions[i]

            #Sense position
            if self.sensing_method == 'model':
                tru, err, is_good = self.sense_model(img, det_var, pos0)
            else:
                tru, err, is_good = self.sense_centroid(img, det_var, pos0)

            #Debugging plot
            if self.debug:
                self.show_plot(img, pos0, tru)

            #Append
            truth = np.concatenate((truth, [tru]))
            errs = np.concatenate((errs, [err]))
            flags = np.concatenate((flags, [is_good]))

        tok = time.perf_counter()
        print(f'Done! Time: {tok-tik:.1f} [s]\n')

        #Save data
        if self.do_save:
            breakpoint()
            save_file = self.image_file.split('.h5')[0].split('/')[-1] + \
                f'__truth_{self.sensing_method}.h5'
            with h5py.File(f'{self.save_dir}/{save_file}', 'w') as f:
                f.create_dataset('truth', data=truth)
                f.create_dataset('errs', data=errs)
                f.create_dataset('flags', data=flags)

############################################
############################################

############################################
####	Model Fitting Algorithm ####
############################################

    def off_rad(self, off):
        #TODO: add ap mask
        return np.sqrt((self.xx - off[0])**2. + (self.yy - off[1])**2.)

    def errfunc(self, pp, yy, ee):
        """Model residual"""
        return (self.fitfunc(pp) - yy)/ee

    def fitfunc(self, pp):
        """Model Image"""
        return pp[2] * j0(self.kRz * self.off_rad(pp[:2]))**2.

    def fitfunc_no_mask(self, pp):
        """Only used for plotting reasons"""
        rad = np.sqrt((self.xx - pp[0])**2. + (self.yy - pp[1])**2.)
        return pp[2] * j0(self.kRz * rad)**2.

    def jacfunc(self, pp, yy, ee):
        """Jacobian"""
        rr = self.off_rad(pp[:2])
        #TODO: add ap_mask to self.xx, self.yy
        #Get rid of zeros to prevent divide by zero in jacobian
        rr[rr == 0.] = 1e-12
        dfda = j0(self.kRz*rr)**2. / ee
        dfdxy = pp[2]*2.*j0(self.kRz*rr)*j1(self.kRz*rr)*self.kRz / rr / ee
        dfdx = (self.xx - pp[0])*dfdxy
        dfdy = (self.yy - pp[1])*dfdxy
        jac = np.vstack((dfdx, dfdy, dfda)).T
        return jac

    def sense_model(self, in_img, det_var, off_guess):

        #Flatten image and add gain
        img = in_img.flatten() * self.ccd_gain

        #Cropped image to exclude aperture mask
        # img = in_img[self.ap_mask]    #TODO: add aperture mask

        #Zero out negative values
        img[img < 0.] = 0.

        #Calculate noise
        noise = np.sqrt(img + det_var)

        #Get initial guess (convert to pixels)
        off0 = off_guess / self.pupil_mag / self.binning
        x0 = [off0[0], off0[1], img.max()]

        ### Solve ###

        #Solve
        out = least_squares(self.errfunc, x0, jac=self.jacfunc,\
            args=(img, noise), method='lm')
        ans = out.x

        #Check for clean exit
        if not out.success:
            return -1*np.ones(2), -1*np.ones(2), False

        #Get output Jacobian matrix
        out_jac = out.jac

        ### Get error estimate from covariance matrix
        fit = self.fitfunc(ans)
        res = fit - img                 #residuals

        #Jacobian
        jac = np.dot(out_jac.T, out_jac)
        try:
            jac_inv = np.linalg.inv(jac)
        except np.linalg.LinAlgError as e:
            jac_inv = np.linalg.pinv(jac)

        #Covariance matrix
        cov = jac_inv * np.dot(res,res) / float(img.size - len(ans))

        #Values to return
        pos = ans[:-1]
        err = np.sqrt(cov.diagonal()[:2])

        #Scale errs for when there is a pupil mask #TODO: kluge
        # errs *= self.obscur_frac**2

        #Convert to physical units [m]
        pos *= self.pupil_mag
        err *= self.pupil_mag

        return pos, err, True

############################################
############################################

############################################
####	Centroid Algorithm ####
############################################

    def sense_centroid(self, in_img, det_var, off_guess):

        #Flatten image and add gain
        img = in_img.flatten()

        #Zero out negative values
        img[img < 0.] = 0.

        #Get thresholded image
        thr_inds = self.get_threshold_inds(img)

        #Threshold
        img_thr = img[thr_inds]
        xx_thr = self.xx[thr_inds]
        yy_thr = self.yy[thr_inds]

        #Get total image counts
        I_tot = img_thr.sum()

        #Get centroid + errors
        xcen, xerr = self.get_centroid_and_errors(img_thr, I_tot, xx_thr, det_var)
        ycen, yerr = self.get_centroid_and_errors(img_thr, I_tot, yy_thr, det_var)

        #Cleanup
        del img_thr, thr_inds, xx_thr, yy_thr

        #Return arrays
        pos = np.array([xcen, ycen])
        err = np.array([xerr, yerr])

        #Convert to physical units [m]
        pos *= self.pupil_mag
        err *= self.pupil_mag

        return pos, err, True

    def get_threshold_inds(self, II):
        #Calculate noise threshold from max of image. Ignore all pixels below threshold
        thr = II.max()*self.cen_threshold

        #Lower threshold if not enough pixels are above threshold
        cntr = 0
        while II[II > thr].size < len(II)/4.  and cntr < 20:
            thr *= 0.9
            cntr += 1

        #Get indices above threshold
        inds = II > thr

        return inds

    def get_centroid_and_errors(self, II, I_tot, rr, det_var):
        #Get centroid
        cen = (rr*II).sum()/I_tot

        #Get centroid error
        err = np.sqrt( (0.25*(II**2.).sum() + (II*rr**2.).sum() + det_var * \
            (rr**2.).sum() + (rr*II).sum()**2./I_tot * (1. + II.size * \
            det_var/II.sum()) )/I_tot**2.)

        return cen, err

############################################
############################################

############################################
####	Misc. ####
############################################

    def show_plot(self, img, pos0, pos):
        print(np.hypot(*(pos0 - pos)))
        #Get image extent
        img_extent = [self.xx[0]*self.pupil_mag, \
            self.xx[-1]*self.pupil_mag, \
            self.yy[-1]*self.pupil_mag, \
            self.yy[0]*self.pupil_mag]

        plt.cla()
        plt.imshow(img.reshape(self.img_shp), extent=img_extent)
        plt.plot(pos[0], pos[1], 'rs')
        plt.plot(pos0[0], pos0[1], 'kx')
        plt.show()
        breakpoint()

############################################
############################################

if __name__ == '__main__':

    params = {
        # 'image_file':       './Results/data_20s_bin4__none__median.h5',
        # 'image_file':       './Results/data_20s_bin4__none__median.h5',
        'image_file':       './Old/Results/data_30s_bin1__none__median.h5',
        'debug':            [False, True][1],
        'do_save':          [False, True][0],
        'sensing_method':   'centroid',
    }

    sen = Truth_Sensor(params)
    sen.get_positions()
