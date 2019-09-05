from torch import nn
from torch.autograd import Variable
import torch
import torch.optim as optim


class classifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(classifier, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.lsm = nn.LogSoftmax(dim=1)

    def forward(self, input):
        output = self.fc(input)
        output = self.lsm(output)
        return output


def pre_train(opt, models, loaders):
    cls = classifier(opt.res_size, opt.split)
    optimizerC = optim.Adam(cls.parameters(), lr=opt.lr_c, betas=(opt.beta1, 0.999))
    if opt.cuda:
        cls.cuda()
    for i in range(opt.pre_train_epoch):
        correct_num = 0
        complete_num = 0
        for images, index, images_class in loaders.img_loader_train:
            cls.zero_grad()
            if opt.cuda:
                images = images.cuda()
                images_class = images_class.cuda()
            res_real = models.ext(images)
            res_real = Variable(res_real, requires_grad=True)
            cls_result = cls(res_real)
            images_class = torch.max(images_class, 1)[1].long()
            cls_loss = models.cls_criterion(cls_result, images_class)
            cls_loss.backward()
            optimizerC.step()
            pred = cls_result.data.max(1)[1]
            correct_num += (pred == images_class).sum()
            complete_num += images_class.shape[0]
        acc = float(correct_num) / float(complete_num)
        print('Pre-Training Acc = {}'.format(acc))
        print('-----------------------------------------------------')
    torch.save(cls, opt.pre_train_path)
    print('Final Pre-Training Acc = {}'.format(acc))
    return cls
