"""
imager.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 04-02-2021

Description: Class that uses the FLYER package to control the camera and motion
    stage in the Princeton starshade testbed to take a number of pupil plane
    images while stepping the camera across the starshade's shadow.

"""

import numpy as np
import flyer
import os
import atexit

class Imager(object):

    def __init__(self, params={}):
        self.set_parameters(params)
        self.load_pilot()

    def set_parameters(self, params):
        def_pms = {
            ### FLYER params ###
            'is_sim':               False,
            'save_dir':             './',
            'do_save':              False,
            'verbose':              False,
            ### Camera params ###
            'exp_time':             1,
            'num_scans':            3,
            'camera_wait_temp':     True,
            'camera_stable_temp':   False,
            'camera_temp':          -40,
            'camera_is_EM':         False,
            'camera_EM_gain':       300,
            'camera_pupil_frame':   None,
            'camera_pupil_center':  None,
            'camera_pupil_width':   None,
            'binning':              1,
            ### Motion params ###
            'rad':                  3,          #[mm]
            'nsteps':               10,
            'dstep':                None,       #[mm]
            'zero_pos':             [0,0],      #[motor steps]
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

        ### Derived ###
        if self.dstep is not None:
            self.steps = np.arange(-self.rad, self.rad + self.dstep, self.dstep)
        else:
            self.steps = np.linspace(-self.rad, self.rad, self.nsteps)

        self.nsteps = len(self.steps)
        self.dstep = self.steps[1] - self.steps[0]
        self.zero_pos = np.atleast_1d(self.zero_pos)

############################################
####    Setup ####
############################################

    def setup_run(self):
        #Start up pilot
        self.start_up_pilot()

        #Create save directory
        self.create_save_dir()

        #Close up shop
        atexit.register(self.pilot.close_up_shop)

    def create_save_dir(self):
        if not self.do_save:
            return

        #Create directory
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)

############################################
############################################

############################################
####    Main Script ####
############################################

    def run_script(self):
        #Setup
        self.setup_run()

        print(f'\n***Running steps from {-self.rad:.1f} to {self.rad:.1f}: ' + \
            f'Nstep = {self.nsteps}, Dstep = {self.dstep:.2f}; ' + \
            f'{self.num_scans}x{self.exp_time} s Exp. ***\n')

        #Loop through x/y
        img_cntr = 0
        for yy in self.steps:
            for xx in self.steps:
                if xx == self.steps.min():
                    print(f'Position: ({xx:.2f}, {yy:.2f})')

                #Move to position
                self.move_to_pos(xx, yy)

                #Take picture
                self.take_picture(img_cntr)

                #Increase counter
                img_cntr += 1

############################################
############################################

############################################
####   Pilot ####
############################################

    def load_pilot(self):
        #get tel rad for proper binning
        trad = {1:1, 2:1.5, 4:3}[self.binning]
        pms = {'is_sim': self.is_sim, 'verbose': self.verbose, \
            'spc_tel_radius': trad}
        for k in self.params.keys():
            if k.startswith('camera'):
                pms[k] = getattr(self, k)

        self.pilot = flyer.Pilot(pms, do_start_up=True, start_hardware=False)
        self.camera = self.pilot.hardware.camera
        self.stage = self.pilot.hardware.stage
        self.camera.verbose = self.verbose
        self.stage.verbose = self.verbose

    def start_up_pilot(self):
        #Start up hardware
        self.camera.start_up(plane='pupil')
        self.stage.start_up()

    def move_to_pos(self, xpos, ypos):
        #Convert to absolution position (need to convert to mm)
        abs_pos = self.zero_pos/self.stage.COUNTS_PER_MM + \
            1e3 * np.array([xpos, ypos])

        #Move
        self.stage.move_abs_position(*abs_pos)

    def take_picture(self, cntr=0):
        #Create filename to save if saving
        if self.do_save:
            fname = os.path.join(self.save_dir, f"image__{str(cntr).zfill(4)}")
        else:
            fname = None

        #Take picture
        return self.camera.take_series(self.exp_time, num_scans=self.num_scans, \
            filename=fname)

############################################
############################################

if __name__ == '__main__':

    params = {
        'is_sim':           True,
        'dstep':            0.1,
        'exp_time':         0.1,
        'nsteps':           10,
        'rad':              2,
        # 'do_save':      True,
        'save_dir_base':    './Results',
        'session_name':     'test',
    }

    imr = Imager(params)
    imr.run_script()
