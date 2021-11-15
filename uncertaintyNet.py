import h5py
import numpy as np
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset, DataLoader


lr = 1e-3
num_epochs = 15


class PredictionDataset(Dataset):
    def __init__(self, model_name, data_run):
        f = h5py.File(os.path.join('Test_Results', f'{data_run}__{model_name}.h5'))
        self.pred_pos = f['pred_pos']
        self.xerr = f['xerr']
        self.yerr = f['yerr']

    def __len__(self):
        return len(self.xerr)

    def __getitem__(self, idx):
        rerr = np.sqrt(self.xerr[idx] ** 2 + self.yerr[idx] ** 2)
        xyerr = np.array([self.xerr[idx], self.yerr[idx]])
        xyerr = np.abs(xyerr)
        # xyerr = 100 * xyerr
        # xyerr = np.multiply(np.log(1000 * np.abs(xyerr)), np.sign(xyerr))
        # xybool = np.array([int(self.xerr[idx] > 0.3 / np.sqrt(2)), int(self.yerr[idx] > 0.3 / np.sqrt(2))])
        return [self.pred_pos[idx].astype(np.float32), xyerr.astype(np.float32)]


class UncertaintyNet(nn.Module):
    def __init__(self):
        super(UncertaintyNet, self).__init__()
        self.fc1 = nn.Linear(2, 7)
        # self.fc2 = nn.Linear(7, 7)
        self.fc3 = nn.Linear(7, 2)

    def forward(self, X):
        X = self.fc1(X)
        # X = self.fc2(X)
        X = self.fc3(X)
        return X

def train(model, trainloader, optimizer, epoch):
    model.train()
    for batch_idx, batch in enumerate(trainloader):
        optimizer.zero_grad()
        output = model(batch[0])
        # print('train')
        # print(output)
        # print(batch[1])
        loss = F.l1_loss(output, batch[1])
        loss.backward()
        optimizer.step()
        if batch_idx % 25 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx*len(batch[1])}/{len(trainloader.dataset)}]\tLoss: {loss.item()/len(batch[1])}')


def test(model, testloader):
    model.eval()
    test_loss = 0
    test_acc = 0
    with torch.no_grad():
        for batch in testloader:
            output = model(batch[0])
            # print('test')
            # print(output)
            # print(batch[1])
            test_loss += F.l1_loss(output, batch[1]).item()

    test_loss /= len(testloader.dataset)
    print(f'\nTest Set: Average Loss {test_loss}\n')


def main():
    # trainset = PredictionDataset(model_name='Wide', data_run='Very_Wide_Noisy_Data')
    trainset = PredictionDataset(model_name='Wide', data_run='Wide_Noisy_Data')
    trainloader = DataLoader(trainset, batch_size=8, shuffle=True)

    testset = PredictionDataset(model_name='Wide', data_run='run__6_01_21__data_1s_bin1__spiders__median')
    # testset = PredictionDataset(model_name='Wide', data_run='Far_Noisy_Data')
    testloader = DataLoader(testset, batch_size=1, shuffle=True)

    model = UncertaintyNet()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-2)

    scheduler = OneCycleLR(optimizer, lr, total_steps=num_epochs)

    for epoch in range(num_epochs):
        train(model, trainloader, optimizer, epoch)
        scheduler.step()

    test(model, testloader)

    save_name = 'Uncertainty'
    save_dir = 'models'
    torch.save(model.state_dict(), os.path.join(save_dir, save_name + '.pt'))

if __name__ == '__main__':
    main()
