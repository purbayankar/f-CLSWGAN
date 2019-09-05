import torch
from torchvision import transforms
from PIL import Image


class init_args:
    def __init__(self):
        self.lr = 0.0001
        self.lr_c = 0.001
        self.beta1 = 0.5
        self.res_size = 256 * 6 * 6
        self.nz_size = 312
        self.att_num = 312
        self.class_num = 200
        self.pre_train_epoch = 150
        self.pre_train_path = './pre_train.pt'
        self.post_train_path = './post_train.pt'
        self.first_epoch = 20
        self.second_epoch = 20
        self.cuda = torch.cuda.is_available()
        self.split = 150
        self.info_path = '/media/rj/860C8FFE0C8FE80F/CUB/CUB_200_2011/images.txt'
        self.root = '/media/rj/860C8FFE0C8FE80F/CUB/CUB_200_2011/images/'
        self.att_path = '/media/rj/860C8FFE0C8FE80F/CUB/CUB_200_2011/attributes/image_attribute_labels.txt'
        self.transform = transforms.Compose(
            [transforms.Resize(256), transforms.RandomCrop(224), transforms.RandomHorizontalFlip(),
             transforms.ToTensor(), transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])])
        self.loader = default_loader
        self.shuffle = True
        self.batch_size = 500
        self.item_num = 11788
        self.lambda1 = 10
        self.cls_path = './cls.pt'
        self.cls_weight = 1
        self.alexnet_path = './alexnet.pt'
        self.len_index = 0
        self.dis_path = './dis.pt'
        self.gen_path = './gen.pt'


def default_loader(path):
    return Image.open(path).convert('RGB')
