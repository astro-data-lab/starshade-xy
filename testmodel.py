"""
    Pass processed images through CNN to extract starshade position.
"""
import numpy as np
import h5py
import os
import torch
import torch.nn.functional as F
from torchvision import transforms
from cnn import CNN
import h5py
import atexit
import time

do_save = [False, True][1]

data_run = 'run__6_01_21__data_1s_bin1__spiders__median'

model_name = 'Newest'

save_ext = ''

#Telescope sizes in Lab and Space coordinates [m] (sets scaling factor)
Dtel_lab = 2.201472e-3
Dtel_space = 2.4
lab2space = Dtel_space / Dtel_lab

#Directories
data_dir = './lab_experiments/processing_data/Processed_Images'
model_dir = './models'
save_dir = 'Test_Results'

#######################

#Open test data file
test_loader = h5py.File(os.path.join(data_dir, data_run + '.h5'), 'r')
atexit.register(test_loader.close)
#Get images, amplitudes, and positions
images = test_loader['images']
amplitudes = test_loader['amplitudes']
positions = test_loader['positions']

#Load model
model = CNN()
mod_file = os.path.join(model_dir, model_name + '.pt')
model.load_state_dict(torch.load(mod_file))
model.eval()

#Transform
transform = transforms.Compose([transforms.ToTensor()])

tik = time.perf_counter()
print(f'Testing {model_name} model...')

#Loop through images and get prediction position
predicted_position = np.zeros((0,2))
difference = np.zeros((0,2))
with torch.no_grad():

    for img, amp, pos in zip(images, amplitudes, positions):

        #Catch error
        if amp == -1:
            predictions = np.concatenate((predictions, [[-1,-1]]))
            continue

        #Normalize image by fit amplitude
        img /= amp

        #Change datatype
        img = img.astype('float32')

        #Transform image
        img = transform(img)
        img = torch.unsqueeze(img, 0)

        #Get solved position
        output = model(img)
        output = output.cpu().detach().numpy().squeeze().astype(float)

        #Compare to truth position (scale truth to space)
        diff = output - pos * lab2space

        #Store
        predicted_position = np.concatenate((predicted_position, [output]))
        difference = np.concatenate((difference, [diff]))

tok = time.perf_counter()
print(f'Done! in {tok-tik:.1f} s')

#Save results
if do_save:
    #Make sure directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    if save_ext != '':
        save_ext = '_' + save_ext

    #Save data
    with h5py.File(os.path.join(save_dir, f'{data_run}__{model_name}{save_ext}.h5'), 'w') as f:
        f.create_dataset('predicted_position', data=predicted_position)
        f.create_dataset('difference', data=difference)
        f.create_dataset('lab2space', data=lab2space)
        f.create_dataset('positions', data=positions)
else:
    breakpoint()
