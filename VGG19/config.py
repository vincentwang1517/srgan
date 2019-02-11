from easydict import EasyDict as edict
import json

config = edict()
config.TRAIN = edict()

## Adam
config.TRAIN.batch_size = 128
config.TRAIN.lr_init = 7*1e-5
config.TRAIN.beta1 = 0.9

config.TRAIN.n_epoch = 250
config.TRAIN.lr_start_decay = 50
config.TRAIN.decay_rate = 0.8
config.TRAIN.decay_every = 10

config.MODEL = edict()
config.MODEL.loaded_model = 'checkpoint/VGG19_transfer_20.npz'

## train set location
config.TRAIN.hr_img_path = '../myresnet/data2017/fer2013_train.csv'

config.VALID = edict()
## test set location
config.VALID.hr_img_path = '../myresnet/data2017/fer2013_test.csv'

def log_config(filename, cfg):
    with open(filename, 'w') as f:
        f.write("================================================\n")
        f.write(json.dumps(cfg, indent=4))
        f.write("\n================================================\n")
