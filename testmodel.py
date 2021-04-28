from astropy.stats import bayesian_blocks
import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from cnn import CNN, StarshadeDataset

transform = transforms.Compose([transforms.ToTensor()])

'''
testset = StarshadeDataset('./data/data96/test.csv', './data/data96/all_data/test', transform=transform)
testloader = DataLoader(testset, batch_size=1, shuffle=True)
'''
testloader = h5py.File('h5data/data_30s_bin1.h5', 'r')
images = testloader['images']
positions = testloader['positions']

model = CNN()
model.load_state_dict(torch.load('models/noisy96_all.pt'))
model.eval()

ct = 0
dist = []
xerr = []
yerr = []
with torch.no_grad():
    '''
    for batch in testloader:
        output = model(batch['image'])
        diff = output - batch['xy']
        dist.append((np.sqrt(diff[0,0]*diff[0,0] + diff[0,1]*diff[0,1])).item())
    '''
    for img, pos in zip(images, positions):
        img = img.astype('float32')
        img = img / np.amax(img)
        img = transform(img)
        img = torch.unsqueeze(img, 0)
        output = model(img)
        diff = output - 1000 * pos
        xerr.append(diff[0, 0])
        yerr.append(diff[0, 1])
        dist.append((np.sqrt(diff[0,0]*diff[0,0] + diff[0,1]*diff[0,1])).item())
        if (np.sqrt(diff[0,0]*diff[0,0] + diff[0,1]*diff[0,1])).item() > 0.1:
            ct += 1

plt.scatter(xerr, yerr, marker = '+')
plt.grid()
plt.title('Scatterplot of Errors (SNR = 3.5)')
plt.xlabel('Error in x (m)')
plt.ylabel('Error in y (m)')
plt.show()
'''

print(ct)
plt.hist(dist, bins=bayesian_blocks(dist))
# plt.xlim([0, 0.1])
plt.ylim([0, 2000])
plt.title('Histogram of Errors (SNR = 5.5)')
plt.xlabel('Error (m)')
plt.ylabel('Frequency')
plt.show()
'''
