import train
import args
import utils
import torch
import train_classifier

print('Preparing Args')

opt = args.init_args()

print('Preparing Loaders')

loaders = utils.loaders(opt=opt)

print('Preparing Attribute DataSet')

att_data = utils.att_set(opt=opt)

print('Preparing Models')

models = utils.init_model(opt=opt)

print('Start Pre-Training')

models.pre_train_cls = train_classifier.pre_train(opt=opt, models=models, loaders=loaders)

print('Start First Training Stage')

train.first_stage_train(opt=opt, models=models, loaders=loaders, att_data=att_data)

torch.save(models.netD, opt.dis_path)
torch.save(models.netG, opt.gen_path)


train.second_stage_train(opt=opt, models=models, loaders=loaders, att_data=att_data)

torch.save(models.post_train_cls, opt.post_train_path)

train.final_test(opt=opt, models=models, loaders=loaders, att_data=att_data)

print('Done!')
