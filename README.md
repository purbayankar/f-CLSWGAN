# f-CLSWGAN

# Introduction

This work follows the idea from Yongqin Xian, Tobias Lorenz, Bernt Schiele, Zeynep Akata. "Feature Generating Networks for Zero-Shot Learning." CVPR (2018). 

I did *not* copy any codes directly, except the calc_gradient_penalty function (about 15 lines) in train.py.

All of the work is developed by myself.

The net sturcture almost the same as f-CLSWGAN. The trianing setting is pretty similar.

# Train the net.

# Environment

Python: 3.7,
PyTorch: 1.2.

## Prepare dataset

Firstly, download the CUB dataset, then edit the 'path' in args.py to point to your dataset location.

Use 'python main.py' to start the training .
