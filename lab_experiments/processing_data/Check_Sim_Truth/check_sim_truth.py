import numpy as np
import h5py
import image_util
import time
import os
import imp

truth_sensor = imp.load_source("truth_sensor", os.path.join(os.pardir, 'truth_sensor.py'))
from truth_sensor import Truth_Sensor

class Sim_Checker(object):

    def __init__(self, params={}):
        self.set_parameters(params)
        self.setup()

############################################
####    Initialization ####
############################################

    def set_parameters(self, params):
        #Default parameters
        def_pms = {
            ### Loading ###
            'base_dir':         './',
            'xtras_dir':        '../../quadrature_code/xtras',
            'session':          'Noisy_Data/sim_check',
            ### Saving ###
            'do_save':          True,
            'do_plot':          False,
            ### Observation ###
            'base_num_pts':     96,
            'image_pad':        10,
            ### Truth Sensor ###
            'image_center':     None,
            'sensing_method':   'model',        #Options: ['model', 'centroid']
            'cen_threshold':    0.75,           #Centroiding threshold
            'wave':             403e-9,
            'ss_radius':        10.1e-3*np.sqrt(680/638),
            'z1':               50.,
            'ccd_dark':         7e-4,           #Dark noise [e/px/s]
            'ccd_read':         3.20,           #Read noise [e/px/frame]
            'ccd_cic':          0.0025,         #CIC noise [e/px/frame]
            'ccd_gain':         0.768,          #inverse gain [ct/e]
        }

        #Set user and default parameters
        for k,v in {**def_pms, **params}.items():
            #Check if valid parameter name
            if k not in def_pms.keys():
                print(f'\nError: Invalid Parameter Name: {k}\n')
                import sys; sys.exit(0)

            #Set parameter value as attribute
            setattr(self, k, v)

        #Directories
        self.load_dir = f'{self.base_dir}/{self.session}'

        #FIXED
        pupil_mag = 1.764
        pixel_size = 13e-6
        self.num_pixel_base = 250

        #Derived
        self.tel_diameter = self.base_num_pts * pupil_mag*pixel_size

    def setup(self):

        #Get image shape
        img = self.get_image(0)
        self.img_shape = img.shape[-2:]

        #Get binning
        self.binning = int(np.round(self.num_pixel_base/self.img_shape[0]))
        self.num_pts = self.base_num_pts // self.binning

        #Load truth sensor
        self.truth = Truth_Sensor(self)

############################################
####    Main Script ####
############################################

    def run_script(self):

        #Startup
        tik = time.perf_counter()
        print('Getting Truth Values...')

        #Get record
        self.record = np.genfromtxt(f"{self.load_dir}/{self.load_dir.split('/')[-1]}.csv", \
            delimiter=',')

        #Get meta
        with h5py.File(f"{self.load_dir}/meta.h5", 'r') as f:
            multi_SNRs = f['multi_SNRs'][()]
            exp_times = f['exp_times'][()]
            num_imgs = f['num_imgs'][()]

        #Loop through steps and get images and exposure times + backgrounds
        locs = np.empty((0, 2))
        for i in range(len(self.record)):

            #Get image
            img = self.get_image(int(self.record[i][0]))
            exp = exp_times[i//num_imgs]

            #Position guess (flip x-sign b/c optics)
            pos0 = self.record[i][1:]

            #Get true position
            pos = self.truth.get_position(img, exp, pos0, do_plot=self.do_plot)

            #Store positions
            locs = np.concatenate((locs, [pos]))

        #Save Data
        if self.do_save:
            fname = f"./Truth_Results/truth__{self.session.split('/')[0]}.h5"
            with h5py.File(fname, 'w') as f:
                f.create_dataset('sim_pos', data=self.record[:,1:])
                f.create_dataset('sensor', data=locs)

        #End
        tok = time.perf_counter()
        print(f'Time: {tok-tik:.1f} [s]\n')

############################################
############################################

############################################
####    Misc Functions ####
############################################

    def get_image(self, num):
        return np.load(f'{self.load_dir}/{str(num).zfill(6)}.npy')

############################################
############################################

if __name__ == '__main__':

    params = {
        # 'do_plot':True,
        'session':          'Sml_Noisy_Data/sim_check',
    }

    chk = Sim_Checker(params)
    chk.run_script()
