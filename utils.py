import torch
import GAN
import torch.optim as optim
from torch.utils.data.dataloader import DataLoader
import scipy.io as sio
import train_classifier


class init_models:
    def __init__(self, opt):
        self.cls = train_classifier.classifier(opt.res_size, opt.class_num)
        self.netD = GAN.Discriminator(opt=opt)
        self.netG = GAN.Generator(opt=opt)
        self.optimizerD = optim.Adam(self.netD.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizerG = optim.Adam(self.netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))
        self.optimizerC = optim.Adam(self.cls.parameters(), lr=opt.lr_c, betas=(opt.beta1, 0.999))
        self.cls_criterion = torch.nn.NLLLoss()
        if opt.cuda:
            self.cls.cuda()
            self.netD.cuda()
            self.netG.cuda()
            self.cls_criterion.cuda()


class images_set_train:
    def __init__(self, opt):
        att_origin = sio.loadmat(opt.att_path)
        res_origin = sio.loadmat(opt.res_path)
        att = att_origin['att']
        res = res_origin['features']
        label = res_origin['labels'] - 1
        loc = att_origin['trainval_loc'].squeeze() - 1

        self.label = torch.from_numpy(label[loc]).long()
        self.res = torch.from_numpy(res[:, loc]).float().T
        self.att = torch.from_numpy(att).float().T
        opt.res_size = self.res.shape[1]
        opt.att_num = self.att.shape[1]
        opt.class_num = self.att.shape[0]
        opt.nz_size = int(opt.res_size * opt.nz_res_ratio)

    def __getitem__(self, index):
        """
        :param index: the index of the res feature.
        :return: res, label, att.
        """
        return self.res[index, :], self.label[index], self.att[self.label[index][0]]

    def __len__(self):
        return self.label.shape[0]


class images_set_test:
    def __init__(self, opt):
        att_origin = sio.loadmat(opt.att_path)
        res_origin = sio.loadmat(opt.res_path)
        att = att_origin['att']
        res = res_origin['features']
        label = res_origin['labels'] - 1
        loc = att_origin['test_unseen_loc'].squeeze() - 1

        self.label = torch.from_numpy(label[loc]).long()
        self.res = torch.from_numpy(res[:, loc]).float().T
        self.att = torch.from_numpy(att).float().T


    def __getitem__(self, index):
        """
        :param index: the index of the res feature.
        :return: res, label, att.
        """
        return self.res[index, :], self.label[index], self.att[self.label[index][0]]

    def __len__(self):
        return self.label.shape[0]


class loaders:
    def __init__(self, opt):
        img_data_train = images_set_train(opt)
        img_data_test = images_set_test(opt)
        self.img_loader_train = DataLoader(img_data_train, batch_size=opt.batch_size, shuffle=opt.shuffle)
        self.img_loader_test = DataLoader(img_data_test, batch_size=opt.batch_size, shuffle=opt.shuffle)
