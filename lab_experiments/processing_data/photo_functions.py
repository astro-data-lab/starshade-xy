"""
photo_functions.py

Author: Anthony Harness
Affiliation: Princeton University
Created on: 05-26-2021

Description: Useful functions for reading photometer data and processing images.

"""

import numpy as np
import h5py
import os
from datetime import datetime, timedelta
from astropy.io import fits
import glob

############################################
#####  Photometer Data #####
############################################

def read_photometer_file(fname, do_save=True, comp_value=2, timezone=4):
    print('\tReading Photometer Data...')

    with open(f'{fname}.dat', 'r') as f:
        lines = f.readlines()

    #Loop through and grab time and values
    tims, vals = [], []
    for ln in lines:
        tnow = np.datetime64(datetime.strptime(ln.split(',')[0], \
            '%m_%d_%y__%H_%M_%S.%f') - timedelta(hours=timezone))
        tims.append(tnow)
        vals.append(float(ln.split(',')[1]))

    photo_time = np.array(tims).copy()
    photo_data = np.array(vals).copy()

    del lines, tims, vals

    #Save
    if do_save:
        with h5py.File(f'{fname}.h5', 'w') as f:
            f.create_dataset('time', data=photo_time.astype('<i8'), \
                compression="gzip", compression_opts=comp_value)
            f.create_dataset('data', data=photo_data, \
                compression="gzip", compression_opts=comp_value)

    return photo_time, photo_data

def load_photometer_h5(fname):
    with h5py.File(f'{fname}.h5', 'r') as f:
        photo_time = f['time'][()].view('<M8[us]').astype(np.datetime64)
        photo_data = f['data'][()]

    return photo_time, photo_data

def load_photometer_data(data_dir, photo_file):
    #Get photo file name
    if photo_file is None:
        flist = glob.glob(os.path.join(data_dir, 'data_*.dat'))
        if len(flist) > 1:
            print('\n!Too many photo files in this directory!\n')
            breakpoint()

        photo_file = os.path.split(flist[0])[-1].split('.dat')[0].split('data_')[-1]

    #full filename
    fname = os.path.join(data_dir, 'data_' + photo_file)

    #Check if we've already loaded into hdf5 file
    if not os.path.exists(f'{fname}.h5'):
        return read_photometer_file(fname)
    else:
        return load_photometer_h5(fname)

def get_photometer_values(photo_time, photo_data, tims, exp, do_plot=False):
    ref_date = np.datetime64('2018-01-01')
    avg_value = []
    for i in range(len(tims)):
        #Get start time
        srt_time = tims[i]
        #Get end time
        end_time = srt_time + np.timedelta64(int(exp*1e6),'us')
        #Get data between start and end
        cur_vals = photo_data[(photo_time >= srt_time) & \
            (photo_time <= end_time)]
        #Check if found data
        if len(cur_vals) == 0:
            #Could mean exp < dtims
            # if exp < (photo_time[1]-photo_time[0]).astype(float)*1e-6:
            if srt_time > photo_time.min() and end_time < photo_time.max():
                #Find closest data point instead
                ind = np.argmin(np.abs(photo_time - srt_time))
                avg_value.append(photo_data[ind-2:ind+2].mean())
                continue
            else:
                print('\nNo photometer data found!\n')
                breakpoint()
        #Save average photo data
        avg_value.append(cur_vals.mean())

        if do_plot:
            import matplotlib.pyplot as plt;plt.ion()
            plt.axvline((srt_time-ref_date).astype(float),color='r')
            plt.axvline((end_time-ref_date).astype(float),color='r')

    if do_plot:
        import matplotlib.pyplot as plt;plt.ion()
        plt.plot((photo_time-ref_date),photo_data)
        breakpoint()
        plt.cla()

    avg_value = np.array(avg_value)

    #Scale by exposure time
    avg_value *= exp

    return avg_value

############################################
############################################

############################################
#####  Image Data #####
############################################

def get_time_data(head, hours_late=9):
    t0 = np.datetime64(head['FRAME'])
    if head['NAXIS3'] == 1:
        time = np.atleast_1d(t0)
    else:
        dt = np.timedelta64(int(head['KCT']*1e6),'us')
        time = t0 + dt*np.arange(head['NAXIS3'])
    #Subtract off bad SOLIS date to put in local time
    time -= np.timedelta64(hours_late,'h')
    return time

def get_header_pkg(head):
    #Get exposure time
    exp = head['EXPOSURE']
    #Get data times
    times = get_time_data(head)
    return exp, times

def load_image(fname):
    with fits.open(fname) as hdu:
        #Get data
        data = hdu[0].data.astype(float)
        #Get header data
        exp, times = get_header_pkg(hdu[0].header)

    return data, exp, times

def get_image_data(fname):
    #Load image
    data, exp, times = load_image(fname)

    return data, exp

############################################
############################################
