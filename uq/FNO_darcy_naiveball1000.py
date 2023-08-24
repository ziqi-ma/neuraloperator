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
train_loader_full, pred_output_encoder_full = get_darcy_loader(data_path="darcy16train1000", sample_start = 0, sample_end = 1000, batch_size=32, shuffle=True, encode_output=True, scale_y=100)
train_loader_half, pred_output_encoder_half = get_darcy_loader(data_path="darcy16train1000", sample_start = 0, sample_end = 500, batch_size=32, shuffle=True, encode_output=True, scale_y=100)
# erroreval_loader_half is just to get the error on second half of training set which will be used to create the other train loader for uncertainty quantifier
erroreval_loader_half, _ = get_darcy_loader(data_path="darcy16train1000", sample_start = 500, sample_end = 1000, batch_size=32, shuffle=True, encode_output=False, scale_y=100)

val_loader, _ = get_darcy_loader(data_path="darcy16val1000", sample_size = 1000, batch_size=32, shuffle=False, encode_output=False, scale_y=100)
test_loader, _ = get_darcy_loader(data_path="darcy16test1000", sample_size = 1000, batch_size=32, shuffle=False,  encode_output=False, scale_y=100)



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
    pred = pred_output_encoder_full.decode(model(x))
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
for idx, sample in enumerate(test_loader):
    x, y = sample['x'], sample['y']
    pred = pred_output_encoder_full.decode(model(x))
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
plt.hist(res16_scores.detach().numpy())
plt.savefig("res16 dist")
print(f"res 16 coverage: {res16_inball_count}/1000 = {res16_inball_count/1000}")


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


## method 3 pointwise train

# first train with half training data
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

trainer.train(train_loader_half, {16:val_loader},
              pred_output_encoder_half,
              model, 
              optimizer,
              scheduler, 
              regularizer=False, 
              training_loss=train_loss,
              eval_losses=eval_losses)

torch.save(model.state_dict(), "model500")
model.eval()

# get error from this model to construct training data for ptwise quantile model
# for test set, first get hat{g}(a)
error_list = []
x_list = []
for idx, sample in enumerate(erroreval_loader_half):
    x, y = sample['x'], sample['y']
    pred = pred_output_encoder_full.decode(model(x))
    error_list.append((pred-y).squeeze())
    x_list.append(x)
second_half_errors = torch.cat(error_list, axis=0)
x_second_half = torch.cat(x_list, axis=0)
del model

# get errors on val and test set now so we don't need this model later
val_error_list = []
val_x_list = []
for idx, sample in enumerate(val_loader):
    x, y = sample['x'], sample['y']
    pred = pred_output_encoder_full.decode(model(x))
    val_error_list.append((pred-y).squeeze())
    val_x_list.append(x)
val_errors = torch.cat(val_error_list, axis=0)
x_val = torch.cat(val_x_list, axis=0)

test_error_list = []
test_x_list = []
for idx, sample in enumerate(test_loader):
    x, y = sample['x'], sample['y']
    pred = pred_output_encoder_full.decode(model(x))
    test_error_list.append((pred-y).squeeze())
    test_x_list.append(x)
test_errors = torch.cat(test_error_list, axis=0)
x_test = torch.cat(test_x_list, axis=0)
del model

# create train loader for the ptwise quantile quantifier
ptwise_quantile_train_loader, quantile_output_encoder = get_darcy_loader_data(datax=x_second_half, datay=second_half_errors, batch_size=32, shuffle=True, encode_output=True) # no scaling y again bc already scaled
val_err_loader, _ = get_darcy_loader_data(datax=x_val, datay=val_errors, batch_size=32, shuffle=False, encode_output=False)
test_err_loader, _ = get_darcy_loader_data(datax=x_test, datay=test_errors, batch_size=32, shuffle=False, encode_output=False)

# load and freeze everything except mlp
model_quantile = TFNO(n_modes=(6,6), hidden_channels=20, projection_channels=16, factorization='tucker', rank=0.42)
model_quantile.load_state_dict(torch.load("model500"))
model_quantile.eval()
for param in model_quantile.parameters():
    param.requires_grad = False 
for param in model_quantile.projection.parameters():
    param.requires_grad = True 

#Create the optimizer
optimizer_quantile = torch.optim.Adam(model_quantile.projection.parameters(), # this should only take unfrozen params
                                lr=3e-3, 
                                weight_decay=1e-4)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=800)


# %%
# Creating the losses
quantile_loss = PointwiseQuantileLoss(quantile=0.9)

train_loss = quantile_loss
eval_losses={'quantile loss': quantile_loss}

# %% 
# Create the trainer
trainer = Trainer(model_quantile, n_epochs=200,
                  device=device,
                  mg_patching_levels=0,
                  wandb_log=False,
                  log_test_interval=3,
                  use_distributed=False,
                  verbose=True)


# %%
# Train on error dataset

trainer.train(ptwise_quantile_train_loader, {16:val_loader},
              quantile_output_encoder,
              model_quantile, 
              optimizer_quantile,
              scheduler, 
              regularizer=False, 
              training_loss=train_loss,
              eval_losses=eval_losses)

torch.save(model.state_dict(), "pt_quantile_model")


# calibrate
# note that in this case we are choosing delta=0.1 (90% CI on function space)
# and m = 1000, so the min alpha (percentage of points in domain that violate ball)
# is sqrt(-ln(0.1)/2000)=0.034. We choose alpha=0.05, i.e. 95% of points in domain lie in ball
# and let t=0.04

# so score is 1-0.05+0.04=99 percentile of all 16*16 points in domain per function
# in terms of |error|/pred_error, i.e. [254]
# and we want the ceil(1001*(0.1-e^-2000*0.04*0.04))/1000 = (1000-59)/1000'th percentile in the 1000 samples
# i.e. ranked, [940]

val_ratio_list = []
for idx, sample in enumerate(val_err_loader):
    x, y = sample['x'], sample['y']
    pred = quantile_output_encoder.decode(model_quantile(x))
    ratio = torch.abs(y)/pred
    val_ratio_list.append(ratio.squeeze())
val_ratios = torch.cat(val_ratio_list, axis=0)

val_ratios_pointwise_quantile = torch.topk(val_ratios.view(val_ratios.shape[0], -1),3, dim=1)
# assert above has shape of [1000,1]
print(val_ratios_pointwise_quantile.shape)
scale_factor = torch.topk(val_ratios_pointwise_quantile, 60, dim=0)
print(scale_factor)

# now scale this and evaluate on test set
# we evaluate both the scaled and unscaled versions
in_unscaled_pred_list = []
in_scaled_pred_list = []
unscaled_avg_interval_list = []
for idx, sample in enumerate(test_err_loader):
    x, y = sample['x'], sample['y']
    pred = quantile_output_encoder.decode(model_quantile(x))
    pred_scaled = pred * scale_factor
    in_unscaled_pred = torch.abs(y) < pred
    in_scaled_pred = torch.abs(y) < pred_scaled

    #we need to get a boolean of whether 95% of pts are in ball
    in_unscaled_pred_flattened = in_unscaled_pred.view(in_unscaled_pred.shape[0], -1)
    in_unscaled_pred_instancewise = in_unscaled_pred_flattened.mean(dim=1) >= 0.95 # expected shape (batchsize, 1)
    print(in_unscaled_pred_instancewise.shape)
    in_unscaled_pred_list.append(in_unscaled_pred_instancewise)

    unscaled_avg_interval = torch.abs(pred).view(pred.shape[0],-1).mean(dim=1)
    unscaled_avg_interval_list.append(unscaled_avg_interval)

    in_scaled_pred_flattened = in_scaled_pred.view(in_scaled_pred.shape[0], -1)
    in_scaled_pred_instancewise = in_scaled_pred_flattened.mean(dim=1) >= 0.95 # expected shape (batchsize, 1)
    in_scaled_pred_list.append(in_scaled_pred_instancewise)

in_unscaled = torch.cat(in_unscaled_pred_list, axis=0)
in_scaled = torch.cat(in_scaled_pred_list, axis=0)
unscaled_intervals = torch.cat(unscaled_avg_interval_list, axis=0)

unscaled_mean_interval = torch.mean(unscaled_intervals, dim=0)
scaled_mean_interval = unscaled_mean_interval * scale_factor

in_unscaled_percentage = torch.mean(in_unscaled, dim=0)
in_scaled_percentage = torch.mean(in_scaled, dim=0)

print(f"Unscaled: {in_unscaled_percentage} of instances satisfy that >= 95% pts drawn are inside the predicted quantile")
print(f"Mean interval width of unscaled is {unscaled_mean_interval}")

print(f"Scaled: {in_scaled_percentage} of instances satisfy that >= 95% pts drawn are inside the predicted quantile")
print("above number is expected to be 90\% - if calibration is done correctly")
print(f"Mean interval width of unscaled is {scaled_mean_interval}")