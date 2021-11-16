import numpy as np
import os
import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F
import time
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from blitz.modules import BayesianLinear
from blitz.utils import variational_estimator

import matplotlib.pyplot as plt

img_size = 116
lr = 1e-3
num_epochs = 5

#Normalization (close to peak suppression / dist_scaling^2)
normalization = 0.03


class StarshadeDataset(Dataset):

    def __init__(self, data_dir, root_name, transform=None):
        self.root_dir = os.path.join(data_dir, root_name)
        self.root_name = root_name
        self.transform = transform
        #Load shifts from csv file
        self.shifts = np.genfromtxt(os.path.join(self.root_dir, \
            root_name + '.csv'), delimiter=',')

    def __len__(self):
        return len(self.shifts)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        #Load image
        img_path = os.path.join(self.root_dir, str(idx).zfill(6) + '.npy')
        image = np.load(img_path).astype('float32')
        #Normalize the image
        image /= normalization

        #Grab the current shift and scale to space-scale
        xy = self.shifts[idx, 1:].astype(np.float32)
        xy *= 1000

        if self.transform:
            image = self.transform(image)

        sample = [image, xy]
        return sample

@variational_estimator
class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, 1)
        self.conv2 = nn.Conv2d(8, 16, 3, 1)
        self.fc1 = nn.Linear(16 * (((img_size - 2) // 2 - 2) // 2) * (((img_size - 2) // 2 - 2) // 2), 128)
        self.fc2 = BayesianLinear(128, 2)

    # def forward(self, X1, X2):
    def forward(self, X1):
        X1 = self.conv1(X1)
        X1 = F.max_pool2d(X1, 2)
        X1 = F.relu(X1)
        X1 = self.conv2(X1)
        X1 = F.max_pool2d(X1, 2)
        X1 = F.relu(X1)
        X1 = torch.flatten(X1, 1)
        # X = torch.cat((X1, X2), dim=1)
        X = X1
        X = self.fc1(X)
        X = F.relu(X)
        X = self.fc2(X)
        return X


def train(model, trainloader, optimizer, epoch):
    model.train()
    for batch_idx, batch in enumerate(trainloader):
        optimizer.zero_grad()
        # output = model(batch[0])
        # loss = F.mse_loss(output, batch[1])
        loss = model.sample_elbo(inputs=batch[0], labels=batch[1], criterion=torch.nn.MSELoss(), sample_nbr=2)
        loss.backward()
        optimizer.step()
        if batch_idx % 25 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx*len(batch[1])}/{len(trainloader.dataset)}]\tLoss: {loss.item()/len(batch[1])}')


def test(model, testloader):
    model.eval()
    test_loss = 0
    with torch.no_grad():
        for batch in testloader:
            # output = model(batch[0], batch[1])
            # test_loss += F.mse_loss(output, batch[2]).item()
            output = model(batch[0])
            test_loss += F.mse_loss(output, batch[1]).item()

    test_loss /= len(testloader.dataset)
    print(f'\nTest Set: Average Loss {test_loss}\n')


def main():

    #Saving
    save_name = 'U2'
    save_dir = 'models'

    #Training
    train_run = 'trainset'
    train_dir_ext = 'Wide_Noisy_Data'

    #Testing
    test_run = 'testset'
    test_dir_ext = train_dir_ext

    ################################

    #Build directories
    data_base_dir = 'quadrature_code/Simulated_Images'
    train_dir = os.path.join(data_base_dir, train_dir_ext)
    test_dir = os.path.join(data_base_dir, test_dir_ext)

    #Transform
    transform = transforms.Compose([transforms.ToTensor()])

    #Load training data
    trainset = StarshadeDataset(train_dir, train_run, transform=transform)
    trainloader = DataLoader(trainset, batch_size=8, shuffle=True)

    #Load testing data
    testset = StarshadeDataset(test_dir, test_run, transform=transform)
    testloader = DataLoader(testset, batch_size=8, shuffle=False)

    #Create model
    model = CNN()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-2)

    #Build scheduler
    scheduler = OneCycleLR(optimizer, lr, total_steps=num_epochs)

    #Loop through epochs
    for epoch in range(num_epochs):
        #Train
        train(model, trainloader, optimizer, epoch)
        #Test
        test(model, testloader)     #TODO: is it necessary to test here?
        #Step scheduler
        scheduler.step()

    #Save model
    torch.save(model.state_dict(), os.path.join(save_dir, save_name + '.pt'))


if __name__ == '__main__':

    #Start timer
    tik = time.perf_counter()

    #Run main script
    main()

    #Print time
    tok = time.perf_counter()
    print(f'\nElapsed time: {tok-tik:.2f} [s]\n')
