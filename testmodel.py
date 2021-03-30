from cnn import CNN, StarshadeDataset
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(0, 1e3)])

testset = StarshadeDataset('./data/test.csv', './noisy_data_10/test', transform=transform)
testloader = DataLoader(testset, batch_size=1, shuffle=True)

model = CNN()
model.load_state_dict(torch.load('noisy_starshade_cnn_10.pt'))
model.eval()

ct = 0
dist = []
with torch.no_grad():
    for batch in testloader:
        output = model(batch['image'])
        diff = output - batch['xy']
        dist.append((np.sqrt(diff[0,0]*diff[0,0] + diff[0,1]*diff[0,1])).item())

bins_list = np.arange(np.max(dist), step=0.01)
plt.hist(dist, bins=bins_list)
plt.title('Histogram of Errors')
plt.xlabel('Error (m)')
plt.ylabel('Frequency')
plt.show()
