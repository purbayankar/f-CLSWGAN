import train
import args
import utils
import train_classifier

print('Preparing Args')

opt = args.init_args()

print('Preparing Loaders')

loaders = utils.loaders(opt=opt)

print('Preparing Models')

models = utils.init_models(opt=opt)

print('Start Pre-Training')

train_classifier.pre_train(opt=opt, models=models, loaders=loaders)

print('Start First Training Stage')

train.first_stage_train(opt=opt, models=models, loaders=loaders)

train.second_stage_train(opt=opt, models=models, loaders=loaders)

train.final_test(opt=opt, models=models, loaders=loaders)

print('Done!')
