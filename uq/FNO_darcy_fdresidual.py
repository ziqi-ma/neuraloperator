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
from mpl_toolkits import mplot3d
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

device = 'cpu'

# use new data with 1000 train +1000 val+1000 test all in 16*16
# and 1000 test in 32*32
train_loader_first_half, initial_model_encoder = get_darcy_loader("darcy16train1000", sample_start = 0, sample_end = 500, batch_size=32, shuffle=True, encode_output=True, scale_y=100)
# train_loader_second_half is just to get the error on second half of training set which will be used to create the other train loader for uncertainty quantifier
train_loader_second_half, _ = get_darcy_loader("darcy16train1000", sample_start = 500, sample_end = 1000, batch_size=32, shuffle=True, encode_output=False, scale_y=100)

val_loader, _ = get_darcy_loader("darcy16val1000", sample_start = 0, sample_end = 1000, batch_size=32, shuffle=False, encode_output=False, scale_y=100)
test_loader, _ = get_darcy_loader("darcy16test1000", sample_start = 0, sample_end = 1000, batch_size=32, shuffle=False,  encode_output=False, scale_y=100)



## method 3 pointwise train

# first train with half training data
def train_initial_model(train_loader, train_encoder, val_loader):
    model = TFNO(n_modes=(6,6), hidden_channels=20, projection_channels=16, factorization="tucker", rank=0.42)
    model = model.to(device)
    n_params = count_params(model)
    print(f'\nOur model has {n_params} parameters.')
    sys.stdout.flush()
    optimizer = torch.optim.Adam(model.parameters(), 
                                lr=3e-3, 
                                weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=800)
    l2loss = LpLoss(d=2, p=2)
    h1loss = H1Loss(d=2)
    train_loss = h1loss
    eval_losses={'h1': h1loss, 'l2': l2loss}
    print('\n### MODEL ###\n', model)
    print('\n### OPTIMIZER ###\n', optimizer)
    print('\n### SCHEDULER ###\n', scheduler)
    print('\n### LOSSES ###')
    print(f'\n * Train: {train_loss}')
    print(f'\n * Test: {eval_losses}')
    sys.stdout.flush()

    trainer = Trainer(model, n_epochs=200,
                  device=device,
                  mg_patching_levels=0,
                  wandb_log=False,
                  log_test_interval=3,
                  use_distributed=False,
                  verbose=True)
    trainer.train(train_loader, {16:val_loader},
              train_encoder,
              model, 
              optimizer,
              scheduler, 
              regularizer=False, 
              training_loss=train_loss,
              eval_losses=eval_losses)
    torch.save(model.state_dict(), "model500")
    model.eval()
    return model

def get_residual(model, encoder, loader):
    error_list = []
    x_list = []
    for idx, sample in enumerate(loader):
        x, y = sample['x'], sample['y']
        pred = encoder.decode(model(x))
        error_list.append(torch.abs(pred-y).squeeze().detach()) # detach, otherwise residual carries gradient of model weight
        x_list.append(x)
    errors = torch.cat(error_list, axis=0)
    xs = torch.cat(x_list, axis=0)
    return xs, errors

def load_initial_model():
    model_quantile = TFNO(n_modes=(6,6), hidden_channels=20, projection_channels=16, factorization='tucker', rank=0.42)
    model_quantile.load_state_dict(torch.load("model500"))
    return model_quantile

def freeze_base_layers(model):
    for param in model.parameters():
        param.requires_grad = False 
    for param in model.projection.parameters():
        param.requires_grad = True 
    return model

def train_quantile_model(base_model, train_loader, encoder, val_loader):
    model_frozen = freeze_base_layers(base_model)
    optimizer_quantile = torch.optim.Adam(model_frozen.projection.parameters(), # this should only take unfrozen params
                                lr=2e-3, 
                                weight_decay=2e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer_quantile, T_max=800)
    quantile_loss = PointwiseQuantileLoss(quantile=0.9)
    train_loss = quantile_loss
    eval_losses={'quantile loss': quantile_loss}
    trainer = Trainer(model_frozen, n_epochs=200,
                  device=device,
                  mg_patching_levels=0,
                  wandb_log=False,
                  log_test_interval=3,
                  use_distributed=False,
                  verbose=True)
    trainer.train_pointwise_err(train_loader, {16:val_loader},
              encoder,
              model_frozen, 
              optimizer_quantile,
              scheduler, 
              regularizer=False, 
              quantile=0.95,
              training_loss=train_loss,
              eval_losses=eval_losses)
    torch.save(model_frozen.state_dict(), "pt_quantile_model")
    return model_frozen

# note that calib loader should output residuals
# calibrate
# note that in this case we are choosing delta=0.01 (99% CI on function space) and alpha=0.1
# and m = 1000, so the min alpha (percentage of points in domain that violate ball)
# is sqrt(-ln(delta)/2m)=sqrt(-ln(0.01)/2000)=0.048. We choose alpha=0.1, i.e. 90% of points in domain lie in ball
# t should < alpha, let t=0.06

# so score is 1-0.1+0.06=94 percentile of all 16*16 points in domain per function
# in terms of |error|/pred_error, i.e. [241]
# and we want the ceil(1001*(0.1-e^-2000*0.06*0.06))/1000 = (1000-101)/1000'th percentile in the 1000 samples
# i.e. ranked, [899]
def calibrate_quantile_model(model, model_encoder, calib_loader):
    val_ratio_list = []
    for idx, sample in enumerate(calib_loader):
        x, y = sample['x'], sample['y']
        pred = model_encoder.decode(model(x)).squeeze()
        ratio = torch.abs(y)/pred
        val_ratio_list.append(ratio.squeeze())
    val_ratios = torch.cat(val_ratio_list, axis=0)
    val_ratios_pointwise_quantile = torch.topk(val_ratios.view(val_ratios.shape[0], -1),15, dim=1).values[:,-1]
    # assert above has shape of [1000,1]
    # print(val_ratios_pointwise_quantile)
    scale_factor = torch.topk(val_ratios_pointwise_quantile, 101, dim=0).values[-1]
    print(scale_factor)
    return scale_factor

def eval_coverage(model, model_encoder, residual_loader, target_point_percentage, scale = 1):
    """
    Get percentage of instances hitting target-percentage pointwise coverage
     (e.g. percenetage of instances with >95% points being covered by quantile model)
     as well as avg band length
    """
    in_pred_list = []
    avg_interval_list = []
    for idx, sample in enumerate(residual_loader):
        x, y = sample['x'], sample['y']
        pred = model_encoder.decode(model(x)) * scale
        in_pred = (torch.abs(y) < pred).float()

        #we need to get a boolean of whether 95% of pts are in ball
        avg_interval = torch.abs(pred).view(pred.shape[0],-1).mean(dim=1)
        avg_interval_list.append(avg_interval)

        in_pred_flattened = in_pred.view(in_pred.shape[0], -1)
        #print(torch.mean(in_pred_flattened,dim=1))
        in_pred_instancewise = torch.mean(in_pred_flattened,dim=1) >= target_point_percentage # expected shape (batchsize, 1)
        in_pred_list.append(in_pred_instancewise.float())

    in_pred = torch.cat(in_pred_list, axis=0)
    intervals = torch.cat(avg_interval_list, axis=0)
    mean_interval = torch.mean(intervals, dim=0)
    in_pred_percentage = torch.mean(in_pred, dim=0)

    print(f"{in_pred_percentage} of instances satisfy that >= {target_point_percentage} pts drawn are inside the predicted quantile")
    print(f"Mean interval width is {mean_interval}")
    return mean_interval, in_pred_percentage


initial_model = load_initial_model()#train_initial_model(train_loader_first_half, initial_model_encoder, val_loader)
initial_model.eval()
x_train_second_half, residual_train = get_residual(initial_model, initial_model_encoder, train_loader_second_half)
x_val, residual_val = get_residual(initial_model, initial_model_encoder, val_loader)
x_test, residual_test = get_residual(initial_model, initial_model_encoder, test_loader)

plot_idx = 17
test_sample = test_loader.dataset[plot_idx]
x_plot, y_plot = test_sample['x'], test_sample['y']
pred_plot = initial_model_encoder.decode(initial_model(torch.unsqueeze(x_plot,0)))
pred_plot = torch.squeeze(pred_plot).detach().numpy()

del initial_model

# do PCA on residual of second half of trainign data, cannot do on val set since that will break exchangeability
flattened_residual = residual_train.reshape(-1, 256)
pca = PCA(n_components=3) # total variance 0.73
pca.fit(flattened_residual)
#coeffs = pca.transform(flattened_residual)

# now on calibration set, get coefficients
flattened_val_res = residual_val.reshape(-1,256)
val_coeffs = pca.transform(flattened_val_res)
# we need the residuals to be simultaneously true, so for 90% CI we need 1-3.33% CI of each
# which is index 16 and 999-16=983
coeff0_up = np.partition(val_coeffs[:,0], 983)[983]
coeff0_lo = np.partition(val_coeffs[:,0], 16)[16]
coeff1_up = np.partition(val_coeffs[:,1], 983)[983]
coeff1_lo = np.partition(val_coeffs[:,1], 16)[16]
coeff2_up = np.partition(val_coeffs[:,2], 983)[983]
coeff2_lo = np.partition(val_coeffs[:,2], 16)[16]
print(coeff0_up)
print(coeff0_lo)
print(coeff1_up)
print(coeff1_lo)
print(coeff2_up)
print(coeff2_lo)

# evaluate on test set
flattened_test_res = residual_test.reshape(-1,256)
test_coeffs = pca.transform(flattened_val_res)
test_in0 = np.logical_and(test_coeffs[:,0]>coeff0_lo, test_coeffs[:,0]<coeff0_up)
test_in1 = np.logical_and(test_coeffs[:,1]>coeff1_lo, test_coeffs[:,1]<coeff1_up)
test_in2 = np.logical_and(test_coeffs[:,2]>coeff2_lo, test_coeffs[:,2]<coeff2_up)
allin = test_in0 & test_in1 & test_in2
allin_ratio = np.mean(allin *1)
print(allin_ratio) #89.9%

# get pointwise width
components = pca.components_ # 3*256

# each point i, upper is each component value * upper if component value>0
# *lower if component value < 0
max0 = np.maximum(components[0,:]*coeff0_lo, components[0,:]*coeff0_up)
min0 = np.minimum(components[0,:]*coeff0_lo, components[0,:]*coeff0_up)
max1 = np.maximum(components[1,:]*coeff0_lo, components[1,:]*coeff0_up)
min1 = np.minimum(components[1,:]*coeff0_lo, components[1,:]*coeff0_up)
max2 = np.maximum(components[2,:]*coeff0_lo, components[2,:]*coeff0_up)
min2 = np.minimum(components[2,:]*coeff0_lo, components[2,:]*coeff0_up)

overall_max = max0+max1+max2
overall_min = min0+min1+min2

bandwidth = overall_max-overall_min
avg_width = np.mean(bandwidth)
print(avg_width)
# this bandwidth is useless bc there could be anything orthogonal to this subspace
# and the variation in those directions coiuld be arbitrarily large

# verify pointwise coverage
print(overall_min)
print("--------")
print(overall_max)
print("!!!!!!!!!!")
print(flattened_test_res.numpy())
above_lo = flattened_test_res.numpy()>overall_min
below_up = flattened_test_res.numpy()<overall_max
pointwise_in = np.logical_and(above_lo, below_up)
print(above_lo)
print(below_up)
allin = pointwise_in.all(axis=1)
pointwise_coverage = np.mean(allin*1)
print(pointwise_coverage)

# plot scaled
# one example
x = np.repeat(np.arange(16)*1.0/16, 16)
y = np.tile(np.arange(16)*1.0/16,16)
z = y_plot.detach().numpy().reshape(256)
pred_up = (pred_plot ).reshape(256) + overall_max
pred_lo = (pred_plot ).reshape(256) + overall_min

# Creating figure
fig = plt.figure(figsize =(14, 9))
ax = plt.axes(projection ='3d')
 
# Creating plot
#ax.plot_surface(x, y, z_residual)
ax.scatter3D(x, y, z, color = "black")
ax.scatter3D(x, y, pred_plot, color = "red")
ax.scatter3D(x, y, pred_up, color = "green")
ax.scatter3D(x, y, pred_lo, color = "yellow")
in_interval = np.mean(np.logical_and(z < pred_up, z > pred_lo) * 1)
plt.title(f"Test index {plot_idx}, {in_interval} points in interval")
plt.savefig("ScatterTrue01.png")


