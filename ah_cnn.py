import numpy as np
import os
import torch
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


img_size = 116
lr = 1e-3
num_epochs = 15
gamma = 0.8


class StarshadeDataset(Dataset):
    
    def __init__(self, data_dir, root_name, transform=None):
        self.root_dir = f'{data_dir}/{root_name}'
        self.root_name = root_name
        self.transform = transform
        self.shifts = np.genfromtxt(f'{self.root_dir}/{root_name}.csv', delimiter=',')

    def __len__(self):
        return len(self.shifts)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = os.path.join(self.root_dir, str(idx).zfill(6) + '.npy')
        image = np.load(img_path).astype('float32')
        image = image / np.amax(image)

        xy = self.shifts[idx, 1:].astype(np.float32)
        xy *= 1000

        import matplotlib.pyplot as plt;plt.ion()
        plt.imshow(image)
        breakpoint()

        if self.transform:
            image = self.transform(image)

        sample = {'image': image, 'xy': xy}
        return sample

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, 1)
        self.conv2 = nn.Conv2d(8, 16, 3, 1)
        self.fc1 = nn.Linear(16 * (((img_size - 2) // 2 - 2) // 2) * (((img_size - 2) // 2 - 2) // 2), 128)
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
            print(f'Train Epoch: {epoch} [{batch_idx*len(batch["xy"])}/{len(trainloader.dataset)}]\tLoss: {loss.item()/len(batch["xy"])}')


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

    #Saving
    save_name = 'test_2'

    #Training
    train_run = 'trainset'
    train_dir = './quadrature_code/Noisy_Data'

    #Testing
    test_run = 'testset'
    test_dir = './quadrature_code/Noisy_Data'

    transform = transforms.Compose([transforms.ToTensor()])

    trainset = StarshadeDataset(train_dir, train_run, transform=transform)
    trainloader = DataLoader(trainset, batch_size=8, shuffle=True)

    testset = StarshadeDataset(test_dir, test_run, transform=transform)
    testloader = DataLoader(testset, batch_size=8, shuffle=False)

    model = CNN()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-2)

    scheduler = StepLR(optimizer, step_size=1, gamma=gamma)
    for epoch in range(num_epochs):
        train(model, trainloader, optimizer, epoch)
        test(model, testloader)
        scheduler.step()

    torch.save(model.state_dict(), f"models/{save_name}.pt")


if __name__ == '__main__':
    main()
