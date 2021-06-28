from astropy.stats import bayesian_blocks
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from ah_cnn import CNN, StarshadeDataset
import h5py

do_save = [False, True][1]

data_run = 'run__6_01_21__data_1s_bin1__spiders__median'

data_dir = './lab_experiments/processing_data/Results'
model_name = 'test'

#######################3

transform = transforms.Compose([transforms.ToTensor()])

testloader = h5py.File(f'{data_dir}/{data_run}.h5', 'r')
images = testloader['images']
positions = testloader['positions']

model = CNN()
model.load_state_dict(torch.load(f'models/{model_name}.pt'))
model.eval()

ct = 0
xerr = np.array([])
yerr = np.array([])
with torch.no_grad():

    for img, pos in zip(images, positions):
        img = img.astype('float32')
        img = img / np.amax(img)
        img = transform(img)
        img = torch.unsqueeze(img, 0)
        output = model(img)
        diff = output - 1000 * pos

        cur_x = diff[0,0].item()
        cur_y = diff[0,1].item()
        cur_r = np.hypot(cur_x, cur_y)

        xerr = np.concatenate((xerr, [cur_x]))
        yerr = np.concatenate((yerr, [cur_y]))

        if cur_r > 0.1:
            ct += 1


if do_save:
    with h5py.File(f'./Results/{data_run}.h5', 'w') as f:
        f.create_dataset('xerr', data=xerr)
        f.create_dataset('yerr', data=yerr)
        f.create_dataset('positions', data=positions)
else:
    breakpoint()
