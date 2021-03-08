from cnn import CNN, StarshadeDataset
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

testset = StarshadeDataset('./data/test.csv', './data/test', transform=transforms.ToTensor())
testloader = DataLoader(testset, batch_size=1, shuffle=False)

model = CNN()
model.load_state_dict(torch.load('starshade_cnn.pt'))
model.eval()

with torch.no_grad():
    for batch in testloader:
        output = model(batch['image'])
        print(np.abs(output - batch['xy']))
