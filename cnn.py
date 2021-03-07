import io
import numpy as np
import os
import pandas as pd
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader


lr = 1e-3
num_epochs = 10
img_size = 74


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

        img_path = os.path.join(self.root_dir, self.shifts.iloc[idx, 0])
        image = io.imread(img_path)
        xy = self.shifts.iloc[idx, 1:]
        xy = np.array([xy])
        sample = {'image': image, 'xy': xy}

        if self.transform:
            sample = self.transform(sample)

        return sample


class CNN(nn.Module):
    def __init__():
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, 1)
        self.conv2 = nn.Conv2d(4, 8, 3, 1)
        self.fc1 = nn.Linear((img_size // 4) * (img_size // 4) * 8, 128)
        self.fc2 = nn.Linear(128, 2)
        
    def forward(self, X):
        X = self.conv1(X)
        X = F.maX_pool2d(X, 2)
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
    for batch_idx, (data, target) in enumerate(train_loader):
        optimizer.zero_grad()
        output = model(data)
        loss = F.mse_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % 25 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx*len(data)}/{len(trainloader.dataset)}]\tLoss: {loss.item()}')
    

def test(model, testloader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for data, target in test_loader:
            output = model(data)
            test_loss += F.mse_loss(output, target).item()
    
    test_loss /= len(testloader.dataset)
    print(f'\nTest Set: Average Loss {test_loss}\n')


def main():

    trainset = StarshadeDataset('train.csv', './data/train')
    trainloader = DataLoader(trainset, batch_size=4, shuffle=True)

    testset = StarshadeDataset('test.csv', './data/test')
    testloader = DataLoader(testset, batch_size=4, shuffle=False)

    model = CNN()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        train(model, trainloader, optimizer, epoch)
        test(model, testloader)

    torch.save(model.state_dict(), "starshade_cnn.pt")


if __name__ == '__main__':
    main()
