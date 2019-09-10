# f-CLSWGAN

# Introduction

This work follows the idea from Yongqin Xian, Tobias Lorenz, Bernt Schiele, Zeynep Akata. "Feature Generating Networks for Zero-Shot Learning." CVPR (2018). 

I did **not** copy any codes directly, except the calc_gradient_penalty function (about 15 lines) in train.py.

All of the work is developed by myself for about 8 hours.

The net sturcture is pretty similar to f-CLSWGAN. The trianing setting is almost the same as it.

# Train the net.

# Environment

* Python: 3.7,

* PyTorch: 1.2,

* scipy.io.

## Prepare dataset

Firstly, download datasets from https://datasets.d2.mpi-inf.mpg.de/xian/xlsa17.zip, then edit the 'res_path' and 'att_path' in args.py to point to your dataset location.

## Start the training.

Use 'python main.py' to start the training .

## BTW

The datasets are 2048-d extracted feature maps from resnet-101.
