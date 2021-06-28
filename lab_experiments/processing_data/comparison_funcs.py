import numpy as np
import h5py
import os
import image_util
import imp

#Import diffraq
diff_dir = os.path.join(os.pardir, os.pardir, "quadrature_code")
diffraq = imp.load_source("diffraq", os.path.join(diff_dir,"diffraq","__init__.py"))

############################################
####    Load Lab Data ####
############################################

def load_lab_data(img_num, sess_run, mask_type, is_med):

    lab_keys = ['num_tel_pts', 'tel_diameter', 'image_pad']
    lab_params = {}

    #Get truth positions
    tru_name = f'{sess_run}__none{["", "__median"][int(is_med)]}'
    with h5py.File(f'./Truth_Results/{tru_name}__truth_model.h5', 'r') as f:
        tru_pos = f['truth'][img_num]

    lab_params['tel_shift'] = tru_pos

    #Load lab data
    lab_name = f'{sess_run}__{mask_type}{["", "__median"][int(is_med)]}'
    with h5py.File(f'./Results/{lab_name}.h5', 'r') as f:

        #Get on-axis image
        lab_img = f['images'][img_num-1]

        #Get params
        for k in lab_keys:
            lab_params[k] = f[k][()]

    #Overwrite params if no mask
    if mask_type == 'none':
        lab_params['tel_diameter'] *= lab_img.shape[-1]/lab_params['num_tel_pts']
        lab_params['num_tel_pts'] = lab_img.shape[-1]
        lab_params['image_pad'] = 0

    return lab_img, lab_params

############################################
############################################

############################################
####    Simulate Image ####
############################################

def simulate_image(lab_params, mask_type):
    #Specify simulation parameters
    params = {

        ### Lab ###
        'wave':             0.403e-6,

        ### Telescope ###
        'with_spiders':     mask_type == 'spiders',
        'skip_mask':        mask_type == 'none',

        ### Starshade ###
        'apod_name':        'lab_ss',
        'num_petals':       12,

        ### Saving ###
        'do_save':          False,
        'verbose':          False,
        'xtras_dir':        os.path.join(diff_dir,"xtras"),

    }

    for k in lab_params.keys():
        params[k] = lab_params[k]

    #Load simulator class
    sim = diffraq.Simulator(params)
    sim.setup_sim()

    #Get diffraction and convert to intensity
    sim_img = np.abs(sim.calculate_diffraction())**2

    return sim_img, sim

############################################
############################################

############################################
####    Normalize ####
############################################

def normalize_images(lab_img, mod_img, pos0, is_max):
    if is_max:
        #Normalize by max
        lab_img /= lab_img.max()
        mod_img /= mod_img.max()

    else:
        #Normalize by sum of center peak
        dn = 15
        cen = (pos0 + np.array(lab_img.shape)/2).astype(int)
        lab_sum = image_util.crop_image(lab_img, cen, dn).sum()
        mod_sum = image_util.crop_image(mod_img, cen, dn).sum()

        lab_img /= lab_sum
        mod_img /= mod_sum

    return lab_img, mod_img

############################################
############################################
