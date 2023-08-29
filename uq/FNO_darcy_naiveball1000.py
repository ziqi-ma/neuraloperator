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
from neuralop.datasets import get_darcy_loader, get_darcy_loader_data
from neuralop.utils import count_params
from neuralop import LpLoss, H1Loss, PointwiseQuantileLoss
from torch.utils.data import Subset
from torch.utils.data import DataLoader
import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from itertools import product

device = 'cpu'

# use new data with 1000 train +1000 val+1000 test all in 16*16
# and 1000 test in 32*32
train_loader_first_half, initial_model_encoder = get_darcy_loader("darcy16train1000", sample_start = 0, sample_end = 500, batch_size=32, shuffle=True, encode_output=True, scale_y=100)
# train_loader_second_half is just to get the error on second half of training set which will be used to create the other train loader for uncertainty quantifier
train_loader_second_half, _ = get_darcy_loader("darcy16train1000", sample_start = 500, sample_end = 1000, batch_size=32, shuffle=True, encode_output=False, scale_y=100)

val_loader, _ = get_darcy_loader("darcy16val1000", sample_start = 0, sample_end = 1000, batch_size=32, shuffle=False, encode_output=False, scale_y=100)
test_loader, _ = get_darcy_loader("darcy16test1000", sample_start = 0, sample_end = 1000, batch_size=32, shuffle=False,  encode_output=False, scale_y=100)



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
                                lr=3e-3, 
                                weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=800)


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
trainer = Trainer(model, n_epochs=200,
                  device=device,
                  mg_patching_levels=0,
                  wandb_log=False,
                  log_test_interval=3,
                  use_distributed=False,
                  verbose=True)


# %%
# Actually train the model on our small Darcy-Flow dataset

trainer.train(train_loader_full, {16:val_loader},
              output_encoder,
              model, 
              optimizer,
              scheduler, 
              regularizer=False, 
              training_loss=train_loss,
              eval_losses=eval_losses)

torch.save(model.state_dict(), "model1000")
model.eval()
'''
model = TFNO(n_modes=(6,6), hidden_channels=20, projection_channels=16, factorization='tucker', rank=0.42)
model.load_state_dict(torch.load("model1000"))
model.eval()

# now do conformal calibration on calibration set
# first predict
score_list = []
for idx, sample in enumerate(val_loader):
    x, y = sample['x'], sample['y']
    pred = initial_model_encoder.decode(model(x))
    diff = (pred - y)**2
    score_list.append(diff.view(diff.shape[0], -1).mean(1))
scores = torch.cat(score_list)

# get empirical percentiles, 5% and 95%, for 1000 samples 50 and 950
score_lo = sorted(scores)[50]
score_hi = sorted(scores)[950]
# plot histogram
print(score_lo)
print(score_hi)
plt.hist(scores.detach().numpy())
plt.savefig("calib dist")

# for test set, first get hat{g}(a)
# resolution 16
res16_inball_count = 0
res16_diff = []
res16_error_list = []
all_y_list = []
for idx, sample in enumerate(test_loader):
    x, y = sample['x'], sample['y']
    all_y_list.append(y.view(y.shape[0], -1))
    pred = initial_model_encoder.decode(model(x))
    diff = (pred - y)**2
    scores = diff.view(diff.shape[0], -1).mean(1)
    in_ball_lo = torch.gt(scores,score_lo)
    in_ball_hi = torch.lt(scores,score_hi)
    res16_error_list.append((pred-y).squeeze())
    res16_diff.append(scores)
    in_ball = torch.logical_and(in_ball_lo, in_ball_hi)
    res16_inball_count += torch.sum(in_ball.int())
res16_scores = torch.cat(res16_diff)
res16_errors = torch.cat(res16_error_list, axis=0)
all_y = torch.cat(all_y_list)
all_y_sd = torch.std(all_y)
print(torch.max(all_y))
print(torch.min(all_y))
y_range = torch.max(all_y)-torch.min(all_y)
plt.hist(res16_scores.detach().numpy())
plt.savefig("res16 dist")
print(f"res 16 coverage: {res16_inball_count}/1000 = {res16_inball_count/1000}")
print(f"y standard deviation overall: {all_y_sd}, overall range: {y_range}")

# method 1 overall conformal
N = 500000

x0 = np.linspace(0, 1, 16)
x1 = np.linspace(0, 1, 16)
kernel = C(0.002, (1e-3, 1e3)) * RBF(0.1, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15)
x0x1 = np.array(list(product(x0, x1)))
y_sample = gp.sample_y(x0x1, N)
scores = (y_sample**2).mean(0).T
print(scores)
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
print(f"true coverage: {true_coverage}") # 0.95
avg_band_len = (pointwise_max - pointwise_min).mean()
print(avg_band_len) # 0.42


## method 2 pointwise conformal
# because of union bound now this needs to be 0.05/256 and 1-0.05/256'th quantile
# i.e. 0.2, we can take [0] and [999]
max2 = torch.topk(res16_errors, 2, dim=0)
pointwise_max = max2.values[0,:,:]
min2 = torch.topk(-res16_errors, 2, dim=0)
pointwise_min = -min2.values[0,:,:]

flags= (res16_errors > torch.Tensor(pointwise_min)) & (res16_errors < torch.Tensor(pointwise_max))
inball = torch.all(flags.view(flags.shape[0], -1), dim=1)
true_coverage = torch.mean(inball.float()).item()
print(f"pointwise calib true coverage: {true_coverage}") # 0.88
avg_band_len = (pointwise_max - pointwise_min).mean()
print(avg_band_len) # 0.33
