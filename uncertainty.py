import atexit
import h5py
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from laplace import Laplace
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

from cnn import CNN, StarshadeDataset   

model_name = 'Wide'

#Directories
data_dir = './lab_experiments/processing_data/Processed_Images'
model_dir = 'models'

# Training
train_run = 'trainset'
train_dir_ext = 'Wide_Noisy_Data'

# Testing
test_run = 'testset'
test_dir_ext = train_dir_ext

# Build directories
data_base_dir = 'quadrature_code/Simulated_Images'
train_dir = os.path.join(data_base_dir, train_dir_ext)
test_dir = os.path.join(data_base_dir, test_dir_ext)

transform = transforms.Compose([transforms.ToTensor()])

# Load training data
trainset = StarshadeDataset(train_dir, train_run, transform=transform)
train_loader = DataLoader(trainset, batch_size=8, shuffle=True)

# Load testing data
testset = StarshadeDataset(test_dir, test_run, transform=transform)
test_loader = DataLoader(testset, batch_size=8, shuffle=False)

# Load pre-trained model
model = CNN()
model.load_state_dict(torch.load(os.path.join(model_dir, model_name + '.pt')))

# User-specified LA flavor
la = Laplace(model, 'regression',
             subset_of_weights='last_layer',
             hessian_structure='full')

# Fit Laplce approximation
la.fit(train_loader)
# la.fit(test_loader)

# Optimize Laplce approximation
n_epochs = 1000
log_prior, log_sigma = torch.ones(1, requires_grad=True), torch.ones(1, requires_grad=True)
hyper_optimizer = torch.optim.Adam([log_prior, log_sigma], lr=1e-1)
for i in range(n_epochs):
    hyper_optimizer.zero_grad()
    neg_marglik = - la.log_marginal_likelihood(log_prior.exp(), log_sigma.exp())
    neg_marglik.backward()
    hyper_optimizer.step()

# Initialize scatter plot lists
r_err = []
uncerts = []

# Evaluate on test set
for batch_idx, batch in enumerate(test_loader):
    f_mu, f_var = la(batch[0])
    f_var = f_var[:, 0, 0]
    f_sigma = f_var.sqrt()
    for i in range(8):
        r_err.append(np.sqrt((f_mu[i][0] - batch[1][i][0]) ** 2 + (f_mu[i][1] - batch[1][i][1]) ** 2).item())
        uncerts.append(f_sigma[i].item())
    # pred_std = torch.sqrt(torch.square(f_sigma) + la.sigma_noise.item() ** 2)

# Plot correlation
plt.scatter(uncerts, r_err, marker='+')
plt.xlabel("uncertainties")
plt.ylabel("r_err")
plt.show()

# x = X_test.flatten().cpu().numpy()
# f_mu, f_var = la(X_test)
# f_mu = f_mu.squeeze().detach().cpu().numpy()
# f_sigma = f_var.squeeze().sqrt().cpu().numpy()
# pred_std = np.sqrt(f_sigma**2 + la.sigma_noise.item()**2)
