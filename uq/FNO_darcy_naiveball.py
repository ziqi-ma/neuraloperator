"""
Training a TFNO on Darcy-Flow
=============================

In this example, we demonstrate how to use the small Darcy-Flow example we ship with the package
to train a Tensorized Fourier-Neural Operator
"""

# %%
# 


import torch
import matplotlib.pyplot as plt
import sys
from neuralop.models import TFNO
from neuralop import Trainer
from neuralop.datasets import load_darcy_flow_small
from neuralop.utils import count_params
from neuralop import LpLoss, H1Loss
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from itertools import product

device = 'cpu'

#darcy dataset has 1000 training samples with res16
#50 test samples with res16
#50 test samples with res32

# take 200 training
# 400 calibration
# 100 (50 res16 50 res32) test
# %%
# Loading the Navier-Stokes dataset in 128x128 resolution


train_loader, test_loaders, output_encoder = load_darcy_flow_small(
        n_train=1000, batch_size=32, 
        test_resolutions=[16, 32], n_tests=[50,50],
        test_batch_sizes=[32, 32],
)
trainset = train_loader.dataset
indices = np.arange(len(trainset))
indices = np.random.permutation(indices)
train_indices = indices [:600]
val_indices = indices[600:1000]

train_subset = Subset(trainset, train_indices)
calibration_subset = Subset(trainset, val_indices)
testset = test_loaders[16].dataset

train_loader_new = DataLoader(train_subset, shuffle=True, batch_size=32)
calibration_loader = DataLoader(calibration_subset, shuffle=False, batch_size=32)
test_loader16_fromtest = DataLoader(testset, shuffle=False, batch_size=32)



# %%
# We create a tensorized FNO model
'''
model = TFNO(n_modes=(6,6), hidden_channels=20, projection_channels=16, factorization="tucker", rank=0.42)
model = model.to(device)

n_params = count_params(model)
print(f'\nOur model has {n_params} parameters.')
sys.stdout.flush()


# %%
#Create the optimizer
optimizer = torch.optim.Adam(model.parameters(), 
                                lr=8e-3, 
                                weight_decay=2e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=300)


# %%
# Creating the losses
l2loss = LpLoss(d=2, p=2)
h1loss = H1Loss(d=2)

train_loss = h1loss
eval_losses={'h1': h1loss, 'l2': l2loss}


# %%
print('\n### MODEL ###\n', model)
print('\n### OPTIMIZER ###\n', optimizer)
print('\n### SCHEDULER ###\n', scheduler)
print('\n### LOSSES ###')
print(f'\n * Train: {train_loss}')
print(f'\n * Test: {eval_losses}')
sys.stdout.flush()


# %% 
# Create the trainer
trainer = Trainer(model, n_epochs=500,
                  device=device,
                  mg_patching_levels=0,
                  wandb_log=False,
                  log_test_interval=5,
                  use_distributed=False,
                  verbose=True)

# %%
# Actually train the model on our small Darcy-Flow dataset

trainer.train(train_loader_new, test_loaders,
              output_encoder,
              model, 
              optimizer,
              scheduler, 
              regularizer=False, 
              training_loss=train_loss,
              eval_losses=eval_losses)

torch.save(model.state_dict(), "model_ckpt")
model.eval()
'''

model = TFNO(n_modes=(6,6), hidden_channels=20, projection_channels=16, factorization='tucker', rank=0.42)
model.load_state_dict(torch.load("model_ckpt"))
model.eval()

# now do conformal calibration on calibration set
# first predict
score_list = []
i = 0
for idx, sample in enumerate(calibration_loader):
    x, y = sample['x'], output_encoder.decode(sample['y'])
    pred = output_encoder.decode(model(x))
    if i == 0:
        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(2, 1, 1)
        ax.imshow(pred[0,:,:].squeeze().detach().numpy())
        ax.set_title('pred')
        ax = fig.add_subplot(2, 1, 2)
        ax.imshow(y[0,:,:].squeeze().detach().numpy())
        ax.set_title('y true')
        fig.savefig("comp")
    i += 1
    diff = (pred - y)**2
    score_list.append(diff.view(diff.shape[0], -1).mean(1))
scores = torch.cat(score_list)


# get empirical percentiles, 5% and 95%, for 400 samples 20 and 380
#print(scores.shape)
score_lo = sorted(scores)[19]
score_hi = sorted(scores)[379]

print(score_lo)
print(score_hi)
print(torch.mean(scores))
# plot histogram
plt.clf()
plt.hist(scores.detach().numpy())
plt.savefig("calib dist")

# for test set, first get hat{g}(a)
# resolution 16

res16_inball_count = 0
res16_diff = []
res16_error_list = []
for idx, sample in enumerate(test_loader16_fromtest):
    x, y = sample['x'], sample['y']
    pred = output_encoder.decode(model(x))
    res16_error_list.append((pred-y).squeeze())
    diff = (pred - y)**2
    scores = diff.view(diff.shape[0], -1).mean(1)
    res16_diff.append(scores)
    in_ball_lo = torch.gt(scores,score_lo)
    in_ball_hi = torch.lt(scores,score_hi)
    in_ball = torch.logical_and(in_ball_lo, in_ball_hi)
    res16_inball_count += torch.sum(in_ball.int())
res16_scores = torch.cat(res16_diff)
res16_errors = torch.cat(res16_error_list, axis=0)
print("------------------------------")
plt.hist(res16_scores.detach().numpy())
plt.savefig("res16-test dist")
print(f"res 16 coverage: {res16_inball_count}/50 = {res16_inball_count/50}")

'''
res32_inball_count = 0
res32_diff = []
for idx, sample in enumerate(test_loaders[32]):
    x, y = sample['x'], sample['y']
    pred = model(x)
    diff = (pred - y)**2
    scores = diff.view(diff.shape[0], -1).mean(1)
    res32_diff.append(scores)
    in_ball_lo = torch.gt(scores,score_lo)
    in_ball_hi = torch.lt(scores,score_hi)
    in_ball = torch.logical_and(in_ball_lo, in_ball_hi)
    res32_inball_count += torch.sum(in_ball.int())
res32_scores = torch.cat(res32_diff)
plt.hist(res32_scores.detach().numpy())
plt.savefig("res32 dist")
'''



#print(f"res 32 coverage: {res32_inball_count}/50 = {res32_inball_count/50}")

# sanity check makes sense, now we want to construct pointwise band
# we want to do this by drawing functions from a gaussian process with mean 0
# this is for error, if its norm is within our thresholds, then choose it
# for N draws, we take the pointwise min and max
N = 500000

x0 = np.linspace(0, 1, 16)
x1 = np.linspace(0, 1, 16)
kernel = C(0.03, (1e-3, 1e3)) * RBF(0.5, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15)
x0x1 = np.array(list(product(x0, x1)))
y_sample = gp.sample_y(x0x1, N)
scores = (y_sample**2).mean(0).T
scores_abovel = scores > score_lo.item()
scores_belowh = scores < score_hi.item()
picked = scores_abovel & scores_belowh
picked_samples = y_sample.T[picked].reshape(16, 16, -1)
print(picked_samples.shape[2])
pointwise_max = picked_samples.max(2)
pointwise_min = picked_samples.min(2)
#print(pointwise_max)
#print(pointwise_min)
error_width = pointwise_max - pointwise_min
plt.clf()
plt.imshow(error_width)
plt.savefig("error_band")

true_inball = 0
total = 0

flags= (res16_errors > torch.Tensor(pointwise_min)) & (res16_errors < torch.Tensor(pointwise_max))
inball = torch.all(flags.view(flags.shape[0], -1), dim=1)
true_coverage = torch.mean(inball.float()).item()
print(f"true coverage: {true_coverage}")
avg_band_len = (pointwise_max - pointwise_min).mean()
print(avg_band_len)
# very sensitive to the gaussian prior, with pointwise all coverage 0.780, avg width
# 0.552

# dumbest thing you can do is to pointwise calibrate, get empirical percentile
# compare the avg band len with this
# because the pointwise thing takes into zero consideration of function, it should be
# bad

# the naive ball is likely also bad
