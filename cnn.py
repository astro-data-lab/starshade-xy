import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from PIL import Image
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


lr = 1e-3
num_epochs = 30
gamma = 0.8


class StarshadeDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        self.shifts = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(self.shifts)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.root_dir, self.shifts.iloc[idx, 0] + '.npy')
        image = np.load(img_path)
        xy = self.shifts.iloc[idx, 1:]
        xy = np.array(xy, dtype=np.float32)
        xy *= 1000

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'xy': xy}
        return sample


class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, 1)
        self.conv2 = nn.Conv2d(8, 16, 3, 1)
        self.fc1 = nn.Linear(4624, 128)
        self.fc2 = nn.Linear(128, 2)
        
    def forward(self, X):
        X = self.conv1(X)
        X = F.max_pool2d(X, 2)
        X = F.relu(X)
        X = self.conv2(X)
        X = F.max_pool2d(X, 2)
        X = F.relu(X)
        X = torch.flatten(X, 1)
        X = self.fc1(X)
        X = F.relu(X)
        X = self.fc2(X)
        return X


def train(model, trainloader, optimizer, epoch):
    model.train()
    for batch_idx, batch in enumerate(trainloader):
        optimizer.zero_grad()
        output = model(batch['image'])
        loss = F.mse_loss(output, batch['xy'])
        loss.backward()
        optimizer.step()
        if batch_idx % 25 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx*len(batch["xy"])}/{len(trainloader.dataset)}]\tLoss: {loss.item()}')
    

def test(model, testloader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch in testloader:
            output = model(batch['image'])
            test_loss += F.mse_loss(output, batch['xy']).item()
    
    test_loss /= len(testloader.dataset)
    print(f'\nTest Set: Average Loss {test_loss}\n')


def main():

    trainset = StarshadeDataset('./data/train.csv', './data/train', transform=transforms.ToTensor())
    trainloader = DataLoader(trainset, batch_size=4, shuffle=True)

    testset = StarshadeDataset('./data/test.csv', './data/test', transform=transforms.ToTensor())
    testloader = DataLoader(testset, batch_size=4, shuffle=False)

    model = CNN()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-2)

    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    for epoch in range(num_epochs):
        train(model, trainloader, optimizer, epoch)
        test(model, testloader)
        scheduler.step()

    torch.save(model.state_dict(), "starshade_cnn.pt")


if __name__ == '__main__':
    main()
