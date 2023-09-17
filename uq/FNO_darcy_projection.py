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
train_loader_full, initial_model_encoder = get_darcy_loader("darcy16train1000", sample_start = 0, sample_end = 1000, batch_size=32, shuffle=True, encode_output=True, scale_y=100)

val_loader, _ = get_darcy_loader("darcy16val1000", sample_start = 0, sample_end = 1000, batch_size=32, shuffle=False, encode_output=False, scale_y=100)
test_loader, _ = get_darcy_loader("darcy16test1000", sample_start = 0, sample_end = 1000, batch_size=32, shuffle=False,  encode_output=False, scale_y=100)


# this is not gonna scale, it's n_train*n_val
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

# this and the naiveball are basically just doing some sort of density estimation
# of the error function, like a test of "is it within 90% of this distribution"
# essentially this is just like kde of the function
# what if we assume it's a gaussian process and just fit a gp?


# on the other hand, the residual no is like fitting another model for error

# for doing the projection, i think if we use the original fno encoder part
# we will get something completely random since the residual is what the original
# basis probably cannot pick up
# so use a diff encoder to project - can we do something like vae to output itself

# train a variational fno to output itself - instead of projection any transformation is ok right?
# this wouldn't work bc we want something not dependent on input, but fno has to have input

# just do a normal fourier basis on 2d, and get dot product? i don't think that will work
# can we use fno somehow? to create a basis? no?

# start simple, just do a normal fourier basis (10) on 2d, do the dot product, now we get a 10d vector
# we do kde on 10d, use that to do conformity score
# now draw new functions from gaussian process with mean 0, dot product with 10 bases, test on conformity score

# above feels unpromising, do pseudo density instead.
# first predict
def get_residual(model, encoder, loader, decode_inputy=False):
    error_list = []
    x_list = []
    for idx, sample in enumerate(loader):
        x, y = sample['x'], sample['y']
        pred = encoder.decode(model(x))
        if decode_inputy:
            y = encoder.decode(y)
        error = (pred-y).squeeze().detach()
        error_list.append(error)
        x_list.append(x)
    errors = torch.cat(error_list, axis=0)
    xs = torch.cat(x_list, axis=0)
    return xs, errors

_, train_errors = get_residual(model, initial_model_encoder, train_loader_full, decode_inputy=True)
_, val_errors = get_residual(model, initial_model_encoder, val_loader)
_, test_errors = get_residual(model, initial_model_encoder, test_loader)

# get pseudo density of val errors based on train errors

def get_gaussian_kernel_value(f1, f2): # f1 and f2 are (m,1) vectors of values on grid pts
    # use l2 norm of functions
    h = 0.1
    mean = 0
    sd = 1
    d = torch.mean((f1-f2)**2)
    kernel_value = (np.pi*sd) * np.exp(-0.5*((d-mean)/sd)**2)
    return kernel_value

def get_pseudo_density_score(f):
    # train_errors is n*16*16
    # f is 16*16
    h = 0.1
    mean = 0
    sd = 1
    diff = f - train_errors # this should be of dim n*16*16
    d = torch.mean(diff, dim=[1,2]) # this should be of dim n*1
    kernel_value = (np.pi*sd) * np.exp(-0.5*((d/h-mean)/sd)**2)# this should be of dim n*1
    score = kernel_value.mean() # it's a one-element array, take it out
    return score

def get_scores(errors):
    scores = []
    for i in range(errors.shape[0]):
        score = get_pseudo_density_score(errors[i,:,:])
        scores.append(score)
    return torch.stack(scores)

# get empirical percentiles, this is nonconformity score so only upper limit
val_scores = get_scores(val_errors)
score_hi = sorted(val_scores)[900]
# plot histogram
print(score_hi)
plt.hist(val_scores)
plt.savefig("calib dist pseudo density nonconformity score")

# for test set
test16_scores = get_scores(test_errors)
test16_in = (test16_scores < score_hi).long().sum()
plt.hist(test16_scores.detach().numpy())
plt.savefig("res16 dist")
print(f"res 16 coverage: {test16_in}/1000 = {test16_in/1000}") #89%

# now get pointwise via GP
# to be implemented
N = 1000

x0 = np.linspace(0, 1, 16)
x1 = np.linspace(0, 1, 16)
kernel = C(0.7, (1e-3, 1e3)) * RBF(0.1, (1e-2, 1e2))
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=15)
x0x1 = np.array(list(product(x0, x1)))
y_sample = gp.sample_y(x0x1, N).T.reshape(-1,16,16)
print(y_sample.shape) # what is the shape? expecting N*16*16
scores = get_scores(torch.Tensor(y_sample))
print(scores.shape)
picked = scores < score_hi.item()
picked_samples = y_sample[picked]
print(picked_samples.shape[0])
pointwise_max = picked_samples.max(0)
pointwise_min = picked_samples.min(0)
#print(pointwise_max)
#print(pointwise_min)
error_width = pointwise_max - pointwise_min
plt.clf()
plt.imshow(error_width)
plt.savefig("error_band")

true_inball = 0
total = 0

flags= (test_errors > torch.Tensor(pointwise_min)) & (test_errors < torch.Tensor(pointwise_max))
inball = torch.all(flags.view(flags.shape[0], -1), dim=1)
true_coverage = torch.mean(inball.float()).item()
print(f"true coverage: {true_coverage}")
avg_band_len = (pointwise_max - pointwise_min).mean()
print(avg_band_len)

# 33%, 4.64, 98%, 5.41