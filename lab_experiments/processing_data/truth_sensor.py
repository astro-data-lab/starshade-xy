"""
truth_sensor.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 06-28-2021

Description: Class that holds functions to extract the position from a pupil image
    via other means such as centroiding or non-linear least squares fit to a model.

"""

import numpy as np
from scipy.special import j0, j1
from scipy.optimize import least_squares

class Truth_Sensor(object):

    def __init__(self, parent):
        self.parent = parent
        self.setup()

############################################
#####  Setup #####
############################################

    def setup(self):
        #Copy over commonly shared
        keys = ['binning', 'cen_threshold', 'ccd_dark', 'ccd_cic', \
            'ccd_read', 'ccd_gain', 'image_center']
        for k in keys:
            setattr(self, k, getattr(self.parent, k))

        #Pupil magnification [m/pixel]
        self.pupil_mag = self.parent.tel_diameter / self.parent.base_num_pts

        #Define "center"
        if self.image_center is None:
            cen = np.array(self.parent.img_shape)/2 + 1
        else:
            cen = self.image_center

        #Get position arrays
        self.yy, self.xx = (np.indices(self.parent.img_shape).T + cen[::-1] - \
            np.array(self.parent.img_shape)).T

        #Flip to match image
        self.yy = self.yy[::-1,::-1]
        self.xx = self.xx[::-1,::-1]

        #Flatten and add binning
        self.xx = self.xx.flatten() * self.binning
        self.yy = self.yy.flatten() * self.binning

        #wavenumber * radius * z [in pixel units]
        self.kRz = 2.*np.pi/self.parent.wave * self.parent.ss_radius * \
            self.pupil_mag / self.parent.z1

        #Sense position
        if self.parent.sensing_method == 'model':
            self.sense_func = self.sense_model
        else:
            self.sense_func = self.sense_centroid

        #Build round pupil mask
        rr = np.hypot(*(np.indices(self.parent.img_shape) - self.parent.img_shape[0]/2))
        self.pupil_mask = np.ones(self.parent.img_shape)
        self.pupil_mask[rr > self.parent.physical_rad_px] = 0
        self.mask_pts = self.pupil_mask.astype(bool).copy().flatten()

############################################
############################################

############################################
####	Main Function ####
############################################

    def get_position(self, img, exp_time, pos0=None, do_plot=False):
        #Get image error
        det_var = self.ccd_dark*exp_time + self.ccd_cic**2. + \
            self.ccd_read**2.

        #Get position guess
        if pos0 is None:
            ind0 = np.argmax(img)
            pos0 = np.array([self.xx[ind0], self.yy[ind0]])*self.pupil_mag

        #Take median image (if not already done)
        if img.ndim == 3:
            img = np.median(img, 0)

        #Get true position
        tru, err, is_good, amp = self.sense_func(img, det_var, pos0)

        #Debug
        if do_plot:
            self.show_plot(img, pos0, tru)

        # #Check flag
        # if not is_good:
        #     print('\n!Bad Truth Position!\n')
        #     breakpoint()

        return tru, amp

############################################
############################################

############################################
####	Model Fitting Algorithm ####
############################################

    def off_rad(self, off):
        return np.sqrt((self.xx - off[0])**2. + (self.yy - off[1])**2.)[self.mask_pts]

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
        #Get rid of zeros to prevent divide by zero in jacobian
        rr[rr == 0.] = 1e-12
        dfda = j0(self.kRz*rr)**2. / ee
        dfdxy = pp[2]*2.*j0(self.kRz*rr)*j1(self.kRz*rr)*self.kRz / rr / ee
        dfdx = (self.xx[self.mask_pts] - pp[0])*dfdxy
        dfdy = (self.yy[self.mask_pts] - pp[1])*dfdxy
        jac = np.vstack((dfdx, dfdy, dfda)).T
        return jac

    def sense_model(self, in_img, det_var, off_guess):

        #Flatten image and add gain
        img = in_img.flatten() * self.ccd_gain

        #Add mask
        img = img[self.mask_pts]

        #Zero out negative values
        img[img < 0.] = 0.

        #Calculate noise
        noise = np.sqrt(img*0 + det_var)

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
            return -1*np.ones(2), -1*np.ones(2), False, -1

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

        #Amplitude
        amp = ans[-1] / self.ccd_gain

        if [False, True][0]:
            #Reshape images
            plt_img = in_img * self.ccd_gain * self.pupil_mask
            plt_fit = self.fitfunc_no_mask(ans).reshape(in_img.shape) * self.pupil_mask

            import matplotlib.pyplot as plt;plt.ion()
            xind,yind = np.unravel_index(np.argmax(plt_img), plt_img.shape)

            # plt.figure()
            plt.cla()
            plt.plot(plt_img[xind], 'b')
            plt.plot(plt_fit[xind],'b--')
            plt.plot(plt_img[:,yind],'r')
            plt.plot(plt_fit[:,yind],'r--')

            breakpoint()

        #Convert to physical units [m]
        pos *= self.pupil_mag
        err *= self.pupil_mag

        return pos, err, True, amp

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

        return pos, err, True, 1

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
        import matplotlib.pyplot as plt;plt.ion()
        print(np.hypot(*(pos0 - pos)))
        #Get image extent
        img_extent = [self.xx[0]*self.pupil_mag, \
            self.xx[-1]*self.pupil_mag, \
            self.yy[-1]*self.pupil_mag, \
            self.yy[0]*self.pupil_mag]

        plt.cla()
        plt.imshow(img, extent=img_extent)
        plt.plot(pos[0], pos[1], 'rs', label='Solved')
        plt.plot(pos0[0], pos0[1], 'kx', label='Guess')
        plt.legend()
        plt.show()
        breakpoint()

############################################
############################################
