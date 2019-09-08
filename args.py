import torch


class init_args:
    def __init__(self):
        self.lr = 0.0001
        self.lr_c = 0.001
        self.beta1 = 0.5
        self.res_size = 2048
        self.nz_size = 312
        self.nz_res_ratio = 1.
        self.att_num = 312
        self.class_num = 200
        self.pre_train_epoch = 100
        self.first_epoch = 2000
        self.second_epoch = 100
        self.cuda = torch.cuda.is_available()
        self.res_path = '/home/rj/DataSet/xlsa17/data/CUB/res101.mat'
        self.att_path = '/home/rj/DataSet/xlsa17/data/CUB/att_splits.mat'
        self.shuffle = True
        self.batch_size = 2000
        self.lambda1 = 10
        self.cls_weight = 1
        self.length = 0
