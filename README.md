# f-CLSWGAN

# Introduction

This work follows the idea from Yongqin Xian, Tobias Lorenz, Bernt Schiele, Zeynep Akata. "Feature Generating Networks for Zero-Shot Learning." CVPR (2018). 

I did **not** copy any codes directly, except the calc_gradient_penalty function (about 15 lines) in train.py.

All of the work is developed by myself for about 8 hours.

The net sturcture is pretty similar to f-CLSWGAN. The trianing setting is almost the same as it.

# Train the net.

# Environment

Python: 3.7,
PyTorch: 1.2.

## Prepare dataset

Firstly, download the CUB dataset, then edit the 'path' in args.py to point to your dataset location.

## Start the training.

Use 'python main.py' to start the training .

## BTW

BTW, I used the pre-trained alexnet, you have to put any pre-trained alexnet under the root directory of the folder.

If you do not have a pre-trained alexnet, just use the random inited alexnet is ok.
