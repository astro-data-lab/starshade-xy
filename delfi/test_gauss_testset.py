import numpy as np
import h5py
import os
import torch
import torch.nn.functional as F
from torchvision import transforms
import h5py
import atexit
import time
import imp
imp_dir = '/home/aharness/repos/starshade-xy'
cnn = imp.load_source("cnn", os.path.join(imp_dir, "cnn_andrew.py"))
from cnn import CNN

do_save = [False, True][1]

data_run = 'n10k_sigd25'

model_name = 'Newest_andrew'

normalization = 0.03
#Telescope sizes in Lab and Space coordinates [m] (sets scaling factor)
Dtel_lab = 2.201472e-3
Dtel_space = 2.4
lab2space = Dtel_space / Dtel_lab

#Directories
model_dir = '/home/aharness/repos/starshade-xy/models'
save_dir = 'Test_Results'

#######################

#Open test data file
test_loader = h5py.File(f'./Sim_Data/{data_run}_data.h5', 'r')
atexit.register(test_loader.close)

#Get images, amplitudes, and positions
images = test_loader['images']
positions = test_loader['positions']

#Load model
model = CNN(images.shape[-1])
mod_file = os.path.join(model_dir, model_name + '.pt')
model.load_state_dict(torch.load(mod_file))
model.eval()

#Transform
transform = transforms.Compose([transforms.ToTensor()])

tik = time.perf_counter()
print(f'Testing {model_name} model with {images.shape[0]} images...')

#Loop through images and get prediction position
predicted_position = np.zeros((0,2))
with torch.no_grad():

    for img, pos in zip(images, positions):

        #Normalize image by
        img /= normalization

        #Change datatype
        img = img.astype('float32')

        #Transform image
        img = transform(img)
        img = torch.unsqueeze(img, 0)

        #Get solved position
        output = model(img)
        output = output.cpu().detach().numpy().squeeze().astype(float)

        #Store
        predicted_position = np.concatenate((predicted_position, [output]))

tok = time.perf_counter()
print(f'Done! in {tok-tik:.1f} s')

#Save results
if do_save:

    # format: x,y,x',y'
    data = np.hstack((positions[()]*lab2space, predicted_position))
    fname = os.path.join(save_dir, f'{data_run}__{model_name}')
    np.save(fname, data)

else:
    breakpoint()
