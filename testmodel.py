import h5py
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from cnn import CNN, StarshadeDataset
import h5py
import atexit
import time

do_save = [False, True][1]

data_run = 'run__6_01_21__data_1s_bin1__spiders__median'
# data_run = 'run__11_15_21_b__data_5s_bin1__spiders__median'

# model_name = 'Wide'
model_name = 'diffnorm'

save_ext = '' + '_' + 'l2s'

#Directories
data_dir = './lab_experiments/processing_data/Processed_Images'
model_dir = 'models'
save_dir = 'Test_Results'

#Normalization (close to peak suppression / dist_scaling^2)
normalization = 0.03

#Lab to space distance conversion
Dtel_lab = 2.201472e-3          #Telescope diameter used in simulations: quadrature_code/generate_images.py
Dtel_space = 2.4                #Roman Telescope
lab2space = Dtel_space / Dtel_lab

#######################

#Open test data file
test_loader = h5py.File(os.path.join(data_dir, data_run + '.h5'), 'r')
atexit.register(test_loader.close)
#Get images and positions
images = test_loader['images']
positions = test_loader['positions']

#Load model
model = CNN()
model.load_state_dict(torch.load(os.path.join(model_dir, model_name + '.pt')))
model.eval()

#Transform
transform = transforms.Compose([transforms.ToTensor()])

tik = time.perf_counter()
print(f'Testing {model_name} model...')

ct = 0
xerr = np.array([])
yerr = np.array([])
predictions = np.zeros((0,2))
with torch.no_grad():

    for img, pos in zip(images, positions):
        img = img.astype('float32')
        #Normalize image
        img /= normalization

        #Transform image
        img = transform(img)
        img = torch.unsqueeze(img, 0)

        #Get solved position
        output = model(img)
        #Compare to truth (after scaling to space-scale)
        diff = output - lab2space * pos

        cur_x = diff[0,0].item()
        cur_y = diff[0,1].item()
        cur_r = np.hypot(cur_x, cur_y)

        xerr = np.concatenate((xerr, [cur_x]))
        yerr = np.concatenate((yerr, [cur_y]))

        #Also save predicted position
        predictions = np.concatenate((predictions, [[output[0,0].item(), output[0,1].item()]]))

        if cur_r > 0.1:
            ct += 1

tok = time.perf_counter()
print(f'Done! in {tok-tik:.1f} s')

#Save results
if do_save:
    #Make sure directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    #Save data
    with h5py.File(os.path.join(save_dir, f'{data_run}__{model_name}{save_ext}.h5'), 'w') as f:
        f.create_dataset('xerr', data=xerr)
        f.create_dataset('yerr', data=yerr)
        f.create_dataset('lab2space', data=lab2space)
        f.create_dataset('predictions', data=predictions)
        f.create_dataset('positions', data=positions)
else:
    breakpoint()
