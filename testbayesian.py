import h5py
import os
import atexit
import torch
from torchvision import transforms
import torch.nn.functional as F
from cnn1 import CNN, StarshadeDataset   

def evaluate_regression(regressor,
                        X,
                        y,
                        samples = 100,
                        std_multiplier = 2):
    preds = [regressor(X) for i in range(samples)]
    preds = torch.stack(preds)
    means = preds.mean(axis=0)
    stds = preds.std(axis=0)
    ci_upper = means + (std_multiplier * stds)
    ci_lower = means - (std_multiplier * stds)
    ic_acc = (ci_lower <= y) * (ci_upper >= y)
    ic_acc = ic_acc.float().mean()
    return ic_acc, (ci_upper >= y).float().mean(), (ci_lower <= y).float().mean()

data_run = 'run__6_01_21__data_1s_bin1__spiders__median'

model_name = 'U'

#Directories
data_dir = './lab_experiments/processing_data/Processed_Images'
model_dir = 'models'
save_dir = 'Test_Results'

#Normalization (close to peak suppression / dist_scaling^2)
normalization = 0.03

#######################

#Open test data file
test_loader = h5py.File(os.path.join(data_dir, data_run + '.h5'), 'r')
atexit.register(test_loader.close)
#Get images and positions
images = test_loader['images']
positions = test_loader['positions']

transform = transforms.Compose([transforms.ToTensor()])

model = CNN()
model.load_state_dict(torch.load(os.path.join(model_dir, model_name + '.pt')))
model.eval()

test_loss = 0
with torch.no_grad():
    for idx, (img, pos) in enumerate(zip(images, positions)):
        img = img.astype('float32')
        #Normalize image
        img /= normalization

        #Transform image
        img = transform(img)
        img = torch.unsqueeze(img, 0)
        pos = torch.tensor([pos])

        output = model(img)
        test_loss += F.mse_loss(output, 1000 * pos).item()
        print(idx)
        # ic_acc, under_ci_upper, over_ci_lower = evaluate_regression(model, img, pos, samples=25, std_multiplier=2)
        # print("CI acc: {:.2f}, CI upper acc: {:.2f}, CI lower acc: {:.2f}".format(ic_acc, under_ci_upper, over_ci_lower))

test_loss /= len(images)

print(f'\nTest Set: Average Loss {test_loss}\n')
