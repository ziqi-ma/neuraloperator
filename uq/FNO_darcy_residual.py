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
        error_list.append((pred-y).squeeze())
        x_list.append(x)
    errors = torch.cat(error_list, axis=0)
    xs = torch.cat(x_list, axis=0)
    return xs, errors

def load_initial_model():
    model_quantile = TFNO(n_modes=(6,6), hidden_channels=20, projection_channels=16, factorization='tucker', rank=0.42)
    model_quantile.load_state_dict(torch.load("model500"))
    model_quantile.eval()
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
                                lr=3e-3, 
                                weight_decay=1e-4)
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
# note that in this case we are choosing delta=0.1 (90% CI on function space)
# and m = 1000, so the min alpha (percentage of points in domain that violate ball)
# is sqrt(-ln(delta)/2m)=sqrt(-ln(0.1)/2000)=0.034. We choose alpha=0.05, i.e. 95% of points in domain lie in ball
# t should < alpha, let t=0.04

# so score is 1-0.05+0.04=99 percentile of all 16*16 points in domain per function
# in terms of |error|/pred_error, i.e. [254]
# and we want the ceil(1001*(0.1-e^-2000*0.04*0.04))/1000 = (1000-59)/1000'th percentile in the 1000 samples
# i.e. ranked, [940]
def calibrate_quantile_model(model, model_encoder, calib_loader):
    val_ratio_list = []
    for idx, sample in enumerate(calib_loader):
        x, y = sample['x'], sample['y']
        pred = model_encoder.decode(model(x))
        ratio = torch.abs(y)/pred
        val_ratio_list.append(ratio.squeeze())
    val_ratios = torch.cat(val_ratio_list, axis=0)

    val_ratios_pointwise_quantile = torch.topk(val_ratios.view(val_ratios.shape[0], -1),3, dim=1)
    # assert above has shape of [1000,1]
    print(val_ratios_pointwise_quantile.shape)
    scale_factor = torch.topk(val_ratios_pointwise_quantile, 60, dim=0)
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
        in_pred = torch.abs(y) < pred

        #we need to get a boolean of whether 95% of pts are in ball
        avg_interval = torch.abs(pred).view(pred.shape[0],-1).mean(dim=1)
        avg_interval_list.append(avg_interval)

        in_pred_flattened = in_pred.view(in_pred.shape[0], -1)
        in_pred_instancewise = in_pred_flattened.mean(dim=1) >= target_point_percentage # expected shape (batchsize, 1)
        print(in_pred_instancewise.shape)
        in_pred_list.append(in_pred_instancewise)

    in_pred = torch.cat(in_pred_list, axis=0)
    intervals = torch.cat(avg_interval_list, axis=0)
    mean_interval = torch.mean(intervals, dim=0)
    in_pred_percentage = torch.mean(in_pred, dim=0)

    print(f"{in_pred_percentage} of instances satisfy that >= {target_point_percentage} pts drawn are inside the predicted quantile")
    print(f"Mean interval width of unscaled is {mean_interval}")
    return mean_interval, in_pred_percentage


initial_model = load_initial_model()#train_initial_model(train_loader_first_half, initial_model_encoder, val_loader)
x_train_second_half, residual_train = get_residual(initial_model, initial_model_encoder, train_loader_second_half)
x_val, residual_val = get_residual(initial_model, initial_model_encoder, val_loader)
x_test, residual_test = get_residual(initial_model, initial_model_encoder, test_loader)

del initial_model

# create train loader for the ptwise quantile quantifier
ptwise_quantile_train_loader, quantile_model_encoder = get_darcy_loader_data(datax=x_train_second_half, datay=residual_train, batch_size=32, shuffle=True, encode_output=True) # no scaling y again bc already scaled
val_err_loader, _ = get_darcy_loader_data(datax=x_val, datay=residual_val, batch_size=32, shuffle=False, encode_output=False)
test_err_loader, _ = get_darcy_loader_data(datax=x_test, datay=residual_test, batch_size=32, shuffle=False, encode_output=False)

reload_model = load_initial_model()
quantile_model = train_quantile_model(reload_model, ptwise_quantile_train_loader, quantile_model_encoder, val_err_loader)
scale_factor = calibrate_quantile_model(quantile_model, quantile_model_encoder, val_err_loader)

# evaluate unscaled
uncalibrated_model_mean_interval, uncalibrated_model_percentage = eval_coverage(quantile_model, quantile_model_encoder, test_err_loader, 0.95)
calibrated_model_mean_interval, calibrated_model_percentage = eval_coverage(quantile_model, quantile_model_encoder, test_err_loader, 0.95, scale=scale_factor)

print(f"uncalibrated, interval {uncalibrated_model_mean_interval}, function space coverage {uncalibrated_model_percentage}")
print(f"calibrated, interval {calibrated_model_mean_interval}, function space coverage {calibrated_model_percentage}")