import tensorflow as tf
import tensorlayer as tl
from tensorlayer.prepro import *
# from config import config, log_config
#
# img_path = config.TRAIN.img_path

import scipy
import numpy as np
import pandas as pd

def get_imgs_fn(file_name, path):
    """ Input an image path and name, return an image array """
    # return scipy.misc.imread(path + file_name).astype(np.float)
    return scipy.misc.imread(path + file_name, mode='RGB')

def crop_sub_imgs_fn(x, is_random=True):
    x = x / (255. / 2.)
    x = x - 1.
    return x

def downsample_fn(x, down_rate=3):
    # We obtained the LR images by downsampling the HR images using bicubic kernel with downsampling factor r = 4.
    down_width = int(x.shape[1] / down_rate)
    down_height = int(x.shape[0] / down_rate) 
    x = imresize(x, size=[down_width, down_height], interp='bicubic', mode=None)
    x = x / (255. / 2.)
    x = x - 1.
    return x

def upsample_fn(x, up_rate=3):
    # We obtained the LR images by downsampling the HR images using bicubic kernel with downsampling factor r = 4.
    up_width = int(x.shape[1] * up_rate)
    up_height = int(x.shape[0] * up_rate) 
    x = imresize(x, size=[up_width, up_height], interp='bicubic', mode=None)
    x = x / (255. / 2.)
    x = x - 1.
    return x

def read_csv_data(file_path, width, height, channel=1):
    print('--- Read Image ---')
    data = pd.read_csv(open(file_path, 'rb')).as_matrix()
    img = np.zeros((len(data), width*height))
    for i in range(len(data)):
        try:
            img[i] = data[i,1].split();
        except:
            print(i)
    img = img.astype('float32')
    img = np.resize(img, (len(data), width, height, channel))
    print(img.shape)
    return img
