from torch import nn
from torch.autograd import Variable


class classifier(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(classifier, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.lsm = nn.LogSoftmax(dim=1)

    def forward(self, inputs):
        output = self.fc(inputs)
        output = self.lsm(output)
        return output


def pre_train(opt, models, loaders):
    for i in range(opt.pre_train_epoch):
        correct_num = 0
        complete_num = 0
        image_class: object
        for res_real, res_real_class, _ in loaders.img_loader_train:
            models.cls.zero_grad()
            if opt.cuda:
                res_real = res_real.cuda()
                res_real_class = res_real_class.cuda()
            res_real = Variable(res_real, requires_grad=True)
            cls_result = models.cls(res_real)
            cls_loss = models.cls_criterion(cls_result, res_real_class.squeeze_())
            cls_loss.backward()
            models.optimizerC.step()
            pred = cls_result.data.max(1)[1]
            correct_num += (pred == res_real_class).sum()
            complete_num += res_real_class.shape[0]
        acc = float(correct_num) / float(complete_num)
        print('Pre-Training Acc = {}'.format(acc))
        print('-----------------------------------------------------')
    print('Final Pre-Training Acc = {}'.format(acc))

