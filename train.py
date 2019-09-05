import utils
import torch
from torch.autograd import Variable
from torch import autograd
import time


def calc_gradient_penalty(opt, netD, res_real, res_fake, att):
    alpha = torch.rand(opt.len_index, 1)
    alpha = alpha.expand(res_real.size())
    if opt.cuda:
        alpha = alpha.cuda()
    interpolates = alpha * res_real + ((1 - alpha) * res_fake)
    interpolates = Variable(interpolates, requires_grad=True)
    disc_interpolates = netD(interpolates, Variable(att))
    ones = torch.ones(disc_interpolates.size())
    if opt.cuda:
        ones = ones.cuda()
    gradients = autograd.grad(outputs=disc_interpolates, inputs=interpolates,
                              grad_outputs=ones,
                              create_graph=True, retain_graph=True, only_inputs=True)[0]
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean() * opt.lambda1
    return gradient_penalty


def first_stage_train(opt, models, loaders, att_data):
    one = torch.tensor(1.)
    mone = torch.tensor(-1.)
    if opt.cuda:
        one = one.cuda()
        mone = mone.cuda()
    for i in range(opt.first_epoch):
        epoch_start_time = time.time()
        print('The {}th epoch starts.'.format(i + 1))
        for image, index, images_class in loaders.img_loader_train:
            opt.len_index = len(index)
            # Train the discriminator.
            for param in models.netD.parameters():
                param.requires_grad = True
            models.netD.zero_grad()
            att = utils.map_atts(index=index, att_data=att_data, opt=opt)
            noise = torch.FloatTensor(opt.len_index, opt.nz_size)
            noise.normal_()
            if opt.cuda:
                images_class = images_class.cuda()
                image = image.cuda()
                noise = noise.cuda()
            res_real = models.ext(image)
            if opt.cuda:
                att = att.cuda()
                noise = noise.cuda()
                res_real = res_real.cuda()
            att_dis = Variable(att, requires_grad=True)
            noise_dis = Variable(noise)
            res_real_dis = Variable(res_real, requires_grad=True)

            dis_real = models.netD(res_real_dis, att_dis)
            dis_real_mean = dis_real.mean()
            dis_real_mean.backward(mone)

            res_fake = models.netG(noise_dis, att_dis)
            dis_fake = models.netD(res_fake, att_dis)
            dis_fake_mean = dis_fake.mean()
            dis_fake_mean.backward(one)

            gradient_penalty = calc_gradient_penalty(opt=opt, netD=models.netD, res_real=res_real, res_fake=res_fake,
                                                     att=att_dis)
            gradient_penalty.backward()
            models.optimizerD.step()

            if i % 3 == 0:
                # Train the generator.
                for param in models.netD.parameters():
                    param.requires_grad = False

                models.netG.zero_grad()

                noise.normal_()
                noise_gen = Variable(noise)
                att_gen = Variable(att, requires_grad=True)
                res_fake = models.netG(noise_gen, att_gen)
                dis_fake = models.netD(res_fake, att_gen)
                dis_fake_mean = - dis_fake.mean()
                cls_result = models.pre_train_cls(res_fake)
                images_class = torch.max(images_class, 1)[1].long()
                cls_loss = models.cls_criterion(cls_result, images_class)

                gen_loss = dis_fake_mean + opt.cls_weight * cls_loss
                gen_loss.backward()
                models.optimizerG.step()

        print('This epoch use {} mins {} secs'.format(int((time.time() - epoch_start_time) / 60),
                                                      int((time.time() - epoch_start_time) % 60)))


def second_stage_train(opt, models, loaders, att_data):
    for param in models.netG.parameters():
        param.requires_grad = False

    for i in range(opt.second_epoch):
        epoch_start_time = time.time()
        print('The {}th epoch starts.'.format(i + 1))
        correct_num = 0
        complete_num = 0
        for _, index, images_class in loaders.img_loader_test:
            opt.len_index = len(index)
            models.pre_train_cls.zero_grad()
            att = utils.map_atts(index=index, att_data=att_data, opt=opt)
            noise = torch.FloatTensor(opt.len_index, opt.nz_size)
            noise.normal_()
            if opt.cuda:
                images_class = images_class.cuda()
                noise = noise.cuda()
                att = att.cuda()
            noise = Variable(noise)
            att = Variable(att, requires_grad=True)
            res_fake = models.netG(noise, att)
            cls_result = models.post_train_cls(res_fake)
            images_class = torch.max(images_class, 1)[1].long()
            cls_loss = models.cls_criterion(cls_result, images_class)
            cls_loss.backward()
            models.optimizerC.step()

            pred = cls_result.data.max(1)[1]
            correct_num += (pred == images_class).sum()
            complete_num += images_class.shape[0]
        acc = float(correct_num) / float(complete_num)
        print('Post-Training Acc = {}'.format(acc))
        print('-----------------------------------------------------')
        print('This epoch use {} mins {} secs'.format(int((time.time() - epoch_start_time) / 60),
                                                      int((time.time() - epoch_start_time) % 60)))
    print('Final Post-Training Acc = {}'.format(acc))


def final_test(opt, models, loaders, att_data):
    correct_num = 0
    complete_num = 0
    for images, index, images_class in loaders.img_loader_test:
        opt.len_index = len(index)
        att = utils.map_atts(index=index, att_data=att_data, opt=opt)
        if opt.cuda:
            images_class = images_class.cuda()
            att = att.cuda()
        att = Variable(att, requires_grad=True)
        res_real = models.ext(images)
        cls_result = models.post_train_cls(res_real)
        images_class = torch.max(images_class, 1)[1].long()
        pred = cls_result.data.max(1)[1]
        correct_num += (pred == images_class).sum()
        complete_num += images_class.shape[0]
    acc = float(correct_num) / float(complete_num)
    print('Final Acc = {}'.format(acc))
