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
import h5py

#Saving
save_name = 'Newest'
save_dir = './models'

#Training
train_run = 'trainset'
train_dir_ext = 'Newest_Data_Noisy'

#Testing
test_run = 'testset'
test_dir_ext = train_dir_ext

#Training parameters
img_size = 116
lr = 1e-3
num_epochs = 3
batch_size = 8

#Normalization (close to peak suppression / calibration mask average)
normalization = 0.03

#Lab to space distance conversion
Dtel_lab = 2.201472e-3          #Telescope diameter used in simulations: quadrature_code/generate_images.py
Dtel_space = 2.4                #Roman Telescope
lab2space = Dtel_space / Dtel_lab

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
        image[image > 9000 * np.median(image)] = 0
        # image /= (0.5 + np.random.rand()) * normalization
        image /= normalization

        #Grab the current shift and scale to space-scale
        xy = self.shifts[idx, 1:].astype(np.float32)
        xy *= lab2space

        if self.transform:
            image = self.transform(image)

        sample = [image, xy]
        return sample

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, 3, 1)
        self.conv2 = nn.Conv2d(8, 16, 3, 1)
        self.conv3 = nn.Conv2d(16, 32, 3, 1)
        self.fc1 = nn.Linear(32 * ((((img_size - 2) // 2 - 2) // 2 - 2) // 2) ** 2, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, X1):
        X1 = self.conv1(X1)
        X1 = F.max_pool2d(X1, 2)
        X1 = F.relu(X1)
        X1 = self.conv2(X1)
        X1 = F.max_pool2d(X1, 2)
        X1 = F.relu(X1)
        X1 = self.conv3(X1)
        X1 = F.max_pool2d(X1, 2)
        X1 = F.relu(X1)
        X1 = torch.flatten(X1, 1)
        X = X1
        X = self.fc1(X)
        X = F.relu(X)
        X = self.fc2(X)
        return X


def train(model, trainloader, optimizer, scheduler, epoch):
    model.train()
    for batch_idx, batch in enumerate(trainloader):
        optimizer.zero_grad()
        # output = model(batch[0], batch[1])
        # loss = F.mse_loss(output, batch[2])
        output = model(batch[0])
        loss = F.mse_loss(output, batch[1])
        loss.backward()
        optimizer.step()
        scheduler.step()
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


def testout(model, testloader):
    model.eval()

    test_loss = 0
    xerr = np.array([])
    yerr = np.array([])
    positions = np.zeros((len(testloader.dataset), 2))
    with torch.no_grad():
        for i, batch in enumerate(testloader):
            # output = model(batch[0], batch[1])
            # test_loss += F.mse_loss(output, batch[2]).item()
            output = model(batch[0])
            diff = output - batch[1]
            cur_x = diff[:,0].detach().numpy()
            cur_y = diff[:,1].detach().numpy()
            xerr = np.concatenate((xerr, cur_x))
            yerr = np.concatenate((yerr, cur_y))
            positions[8*i:8*i+8] = batch[1]

            test_loss += F.mse_loss(output, batch[1]).item()

    test_loss /= len(testloader.dataset)
    print(f'\nTest Set: Average Loss {test_loss}\n')


def main():

    #Build directories
    data_base_dir = 'quadrature_code/Simulated_Images'
    train_dir = os.path.join(data_base_dir, train_dir_ext)
    test_dir = os.path.join(data_base_dir, test_dir_ext)

    #Transform
    transform = transforms.Compose([transforms.ToTensor()])

    #Load training data
    trainset = StarshadeDataset(train_dir, train_run, transform=transform)
    trainloader = DataLoader(trainset, batch_size=batch_size, shuffle=True)

    #Load testing data
    testset = StarshadeDataset(test_dir, test_run, transform=transform)
    testloader = DataLoader(testset, batch_size=batch_size, shuffle=False)

    #Create model
    model = CNN()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-2)

    #Build scheduler
    scheduler = OneCycleLR(optimizer, lr, steps_per_epoch=len(trainloader.dataset) // batch_size + 1, epochs=num_epochs)

    #Loop through epochs
    for epoch in range(num_epochs):
        #Train
        train(model, trainloader, optimizer, scheduler, epoch)
        #Test
        test(model, testloader)     #TODO: is it necessary to test here?
        #Step scheduler
        # scheduler.step()

    testout(model, testloader)
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
