import scipy.io
import torch
mat = scipy.io.loadmat('data/darcy_data/darcy16val1000u.mat')
utensor = torch.tensor(mat["u_mat"])
torch.save(utensor, "data/darcy_data/darcy16val1000u.pt")