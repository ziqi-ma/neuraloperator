"""
A simple Darcy-Flow dataset
===========================
In this example, we demonstrate how to use the small Darcy-Flow example we ship with the package
"""

# %%
# Import the library
# ------------------
# We first import our `neuralop` library and required dependencies.

import matplotlib.pyplot as plt
from neuralop.datasets import load_darcy_flow_small
import torch
# %%
# Load the dataset
# ----------------
# Training samples are 16x16 and we load testing samples at both 
# 16x16 and 32x32 (to test resolution invariance).

train_loader, test_loaders, output_encoder = load_darcy_flow_small(
        n_train=1000, batch_size=4, 
        test_resolutions=[16, 32], n_tests=[200, 200], test_batch_sizes=[4, 2],
        )

train_dataset = train_loader.dataset
test_dataset = test_loaders[16].dataset

# %%
# Visualizing the data
# --------------------
'''
i = 0
mean_list = []
for idx, sample in enumerate(train_loader):
    x, y = sample['x'], sample['y']
    xmean = torch.abs(x.view(x.shape[0], -1)).mean(1)
    ymean = torch.abs(y.view(y.shape[0], -1)).mean(1)
    ymax = torch.abs(y.view(y.shape[0], -1)).max(1)
    if i%10 == 0:
        #print(xmean)
        print(ymean)
    mean_list.append(ymean)
    i += 1
print(torch.mean(torch.cat(mean_list)))

mean_list = []
print("----------------------------")
for idx, sample in enumerate(test_loaders[32]):
    x, y = sample['x'], sample['y']
    xmean = torch.abs(x.view(x.shape[0], -1)).mean(1)
    ymean = torch.abs(y.view(y.shape[0], -1)).mean(1)
    ymax = torch.abs(y.view(y.shape[0], -1)).max(1)
    #print(xmean)
    mean_list.append(ymean)
    print(ymean)
print(torch.mean(torch.cat(mean_list)))
'''
'''
for res, test_loader in test_loaders.items():
    print('res')
    test_data = train_dataset[0]
    x = test_data['x']
    y = test_data['y']

    print(f'Testing samples for res {res} have shape {x.shape[1:]}')


data = train_dataset[0]
x = data['x']
y = data['y']

print(f'Training sample have shape {x.shape[1:]}')


def view_data(data, name):
    x = data['x']
    y = data['y']
    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(2, 2, 1)
    ax.imshow(x[0], cmap='gray')
    ax.set_title('input x')
    ax = fig.add_subplot(2, 2, 2)
    ax.imshow(y.squeeze())
    ax.set_title('input y')
    ax = fig.add_subplot(2, 2, 3)
    ax.imshow(x[1])
    ax.set_title('x: 1st pos embedding')
    ax = fig.add_subplot(2, 2, 4)
    ax.imshow(x[2])
    ax.set_title('x: 2nd pos embedding')
    fig.suptitle('Visualizing one input sample', y=0.98)
    plt.tight_layout()
    fig.savefig(f"visualize/visualize-{name}")


view_data(train_dataset[0], "train0")
view_data(train_dataset[10], "train10")
view_data(train_dataset[301], "train301")
view_data(train_dataset[501], "train501")
view_data(train_dataset[701], "train5=701")

view_data(test_dataset[0], "test0")
view_data(test_dataset[10], "test10")
view_data(test_dataset[31], "test31")
'''

