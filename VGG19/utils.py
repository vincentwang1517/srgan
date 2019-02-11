import tensorflow as tf
import tensorlayer as tl
import random
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
    # x = crop(x, wrg=side, hrg=side, is_random=is_random)
    x = x / (255. / 2.)
    x = x - 1.
    # x = (x - 0.5)*2
    return x

def downsample_fn(x, down_rate=3):
    # We obtained the LR images by downsampling the HR images using bicubic kernel with downsampling factor r = 4.
    down_width = int(x.shape[1] / down_rate)
    down_height = int(x.shape[0] / down_rate) 
    x = imresize(x, size=[down_width, down_height], interp='bicubic', mode=None)
    x = x / (255. / 2.)
    x = x - 1.
    # x = (x - 0.5)*2
    return x

def upsample_fn(x, up_rate=3):
    # We obtained the LR images by downsampling the HR images using bicubic kernel with downsampling factor r = 4.
    up_width = int(x.shape[1] * up_rate)
    up_height = int(x.shape[0] * up_rate) 
    x = imresize(x, size=[up_width, up_height], interp='bicubic', mode=None)
    x = x / (255. / 2.)
    x = x - 1.
    # x = (x - 0.5)*2
    return x

def read_csv_data(file_path, width, height, channel=1):
    print('--- Read Image ---')
    data = pd.read_csv(open(file_path, 'rb')).as_matrix()
    img = np.zeros((len(data), width*height))
    labels = np.zeros((len(data)))
    for i in range(len(data)):
        try:
            img[i] = data[i,1].split();
        except:
            print(i)
        labels[i] = data[i,0]
    img = img.astype('float32')
    img = np.resize(img, (len(data), width, height, channel))
    print(img.shape)
    return labels, img

def shuffle(x, y):
    randomize = np.arange(len(x));
    np.random.shuffle(randomize);
    return x[randomize], y[randomize];

def ImageDataGenerator(img):
    M_rotate = tl.prepro.affine_rotation_matrix(angle=(-20,20))
    M_flip = tl.prepro.affine_horizontal_flip_matrix(prob=0.5)
    M_shift = tl.prepro.affine_shift_matrix(wrg=(-0.2,0.2), hrg=(-0.2,0.2), h=48, w=48)
    M_shear = tl.prepro.affine_shear_matrix(x_shear=(-0.2,0.2), y_shear=(-0.2,0.2))

    M_combined = M_shift.dot(M_shear).dot(M_flip).dot(M_rotate)
    transform_matrix = tl.prepro.transform_matrix_offset_center(M_combined, x=48, y=48)
    result = tl.prepro.affine_transform_cv2(img, transform_matrix)
    return result