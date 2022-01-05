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

do_save = [False, True][1]


model_name = 'diffnorm2'

#Directories
data_dir = './lab_experiments/processing_data/Processed_Images'
model_dir = 'models'
save_dir = 'Test_Results'

#######################

test_run = 'testset'
test_dir_ext = 'Wide_Noisy_Data'

################################

#Build directories
data_base_dir = 'quadrature_code/Simulated_Images'
test_dir = os.path.join(data_base_dir, test_dir_ext)

#Transform
transform = transforms.Compose([transforms.ToTensor()])

#Load testing data
testset = StarshadeDataset(test_dir, test_run, transform=transform)
testloader = DataLoader(testset, batch_size=1, shuffle=False)

#Load model
model = CNN()
model.load_state_dict(torch.load(os.path.join(model_dir, model_name + '.pt')))
model.eval()

ct = 0
xerr = np.array([])
yerr = np.array([])
positions = np.zeros((len(testloader.dataset), 2))
with torch.no_grad():

    for i, batch in enumerate(testloader):
        #Get solved position
        output = model(batch[0])
        #Compare to truth (after scaling to space-scale)
        diff = output - batch[1]

        cur_x = diff[0,0].item()
        cur_y = diff[0,1].item()
        cur_r = np.hypot(cur_x, cur_y)

        xerr = np.concatenate((xerr, [cur_x]))
        yerr = np.concatenate((yerr, [cur_y]))
        positions[i] = batch[1].flatten()

        if cur_r > 0.1:
            ct += 1

#Save results
if do_save:
    #Make sure directory exists
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    #Save data
    with h5py.File(os.path.join(save_dir, f'{test_dir_ext}__{model_name}.h5'), 'w') as f:
        f.create_dataset('xerr', data=xerr)
        f.create_dataset('yerr', data=yerr)
        f.create_dataset('positions', data=positions)
else:
    breakpoint()
