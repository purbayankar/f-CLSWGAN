import torch
import alexnet
import GAN
import torch.optim as optim
import os
from numpy import loadtxt
from PIL import Image
from torch.utils.data.dataloader import DataLoader
import train_classifier


class init_model:
    def __init__(self, opt):
        model = torch.load(opt.alexnet_path)
        self.pre_train_cls = None
        self.post_train_cls = train_classifier.classifier(opt.res_size, opt.class_num - opt.split)
        self.ext = alexnet.extractor()
        self.ext.features.load_state_dict(model.features.state_dict())
        self.ext.avgpool.load_state_dict(model.avgpool.state_dict())
        self.netD = GAN.Discriminator(opt=opt)
        self.netG = GAN.Generator(opt=opt)
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizerC = optim.Adam(self.post_train_cls.parameters(), lr=opt.lr_c, betas=(opt.beta1, 0.999))
        self.cls_criterion = torch.nn.NLLLoss()
        for param in self.ext.parameters():
            param.requires_grad = False
        if opt.cuda:
            self.post_train_cls.cuda()
            self.ext.cuda()
            self.netD.cuda()
            self.netG.cuda()
            self.cls_criterion.cuda()
        del model


def default_loader(path):
    return Image.open(path).convert('RGB')


class images_set_train:
    def __init__(self, opt):
        image_info = open(opt.info_path)
        images_path = []
        images_class_tmp = []
        for line in image_info.readlines():
            line_tmp = line.split()
            image_path = line_tmp[1]
            image_path = os.path.join(opt.root, image_path)
            images_path.append(image_path)
            images_class_tmp.append(os.path.split(image_path)[0])

        # Get the ont-hot classes list for images.
        images_class = torch.zeros(len(images_class_tmp), opt.class_num)
        col = 0
        for split_loc in range(len(images_class_tmp)):
            if split_loc == 0:
                images_class[split_loc][col] = 1
            else:
                if images_class_tmp[split_loc] == images_class_tmp[split_loc - 1]:
                    images_class[split_loc][col] = 1
                else:
                    if col < opt.split - 1:
                        col += 1
                        images_class[split_loc][col] = 1
                    else:
                        break

        self.images_class_train = images_class[:split_loc, :opt.split]
        self.root = opt.root
        self.images_path_train = images_path[:split_loc]
        self.transform = opt.transform
        self.loader = opt.loader

    def __getitem__(self, index):
        """
        :param index: the index of the images.
        :return: first item - index for that image, second item - the image data.
        """
        image_path = self.images_path_train[index]
        image = self.loader(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, index, self.images_class_train[index]

    def __len__(self):
        return len(self.images_path_train)


class images_set_test:
    def __init__(self, opt):
        image_info = open(opt.info_path)
        images_path = []
        images_class_tmp = []
        for line in image_info.readlines():
            line_tmp = line.split()
            image_path = line_tmp[1]
            image_path = os.path.join(opt.root, image_path)
            images_path.append(image_path)
            images_class_tmp.append(os.path.split(image_path)[0])

        # Get the ont-hot classes list for images.
        images_class = torch.zeros(len(images_class_tmp), opt.class_num)
        col = opt.class_num - 1
        for split_loc in range(len(images_class_tmp) - 1, -1, -1):
            if split_loc == len(images_class_tmp) - 1:
                images_class[split_loc][col] = 1
            else:
                if images_class_tmp[split_loc] == images_class_tmp[split_loc + 1]:
                    images_class[split_loc][col] = 1
                else:
                    if col > opt.split:
                        col -= 1
                        images_class[split_loc][col] = 1
                    else:
                        break
        self.images_class_test = images_class[split_loc + 1:, opt.split:]
        self.root = opt.root
        self.images_path_test = images_path[split_loc + 1:][:]
        self.transform = opt.transform
        self.loader = opt.loader

    def __getitem__(self, index):
        """
        :param index: the index of the images.
        :return: first item - index for that image, second item - the image data.
        """
        image_path = self.images_path_test[index]
        image = self.loader(image_path)
        if self.transform is not None:
            image = self.transform(image)
        return image, index, self.images_class_test[index]

    def __len__(self):
        return len(self.images_path_test)


def att_set(opt):
    att_data_ori = loadtxt(opt.att_path)
    att_data_ori = torch.from_numpy(att_data_ori)
    att_data = torch.zeros(opt.item_num, opt.att_num)
    for i in range(opt.item_num):
        for j in range(opt.att_num):
            att_data[i][j] = att_data_ori[i * opt.att_num + j][2]
    if opt.cuda:
        att_data = att_data.cuda()
    return att_data


class loaders:
    def __init__(self, opt):
        img_data_train = images_set_train(opt)
        img_data_test = images_set_test(opt)
        self.img_loader_train = DataLoader(img_data_train, batch_size=opt.batch_size, shuffle=opt.shuffle)
        self.img_loader_test = DataLoader(img_data_test, batch_size=opt.batch_size, shuffle=opt.shuffle)


def map_atts(index, att_data, opt):
    att = torch.FloatTensor(opt.len_index, opt.att_num)
    if opt.cuda:
        att = att.cuda()
    for i in range(len(index)):
        att[i][:] = att_data[index[i]][:]
    return att
