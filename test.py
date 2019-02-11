#! /usr/bin/python
# -*- coding: utf8 -*-

import os, time, pickle, random, time, sys
from datetime import datetime
import numpy as np
from time import localtime, strftime
import logging, scipy
import tensorflow as tf
import tensorlayer as tl
from model import SRGAN_g, SRGAN_d, Vgg19_simple_api
from utils import *
from config import config, log_config

###====================== HYPER-PARAMETERS ===========================###
## Adam
batch_size = config.TRAIN.batch_size
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1
## initialize G
n_epoch_init = config.TRAIN.n_epoch_init
## adversarial learning (SRGAN)
n_epoch = config.TRAIN.n_epoch
lr_decay = config.TRAIN.lr_decay
decay_every = config.TRAIN.decay_every

ni = int(np.sqrt(9))

def evaluate():
    ## create folders to save result images
    save_dir = "samples/valid"
    tl.files.exists_or_mkdir(save_dir)
    checkpoint_dir = "checkpoint"

    ###====================== PRE-LOAD DATA and SAVE SAMPLEs ===========================###

    valid_hr_imgs = read_csv_data(config.VALID.hr_img_path, width=48, height=48, channel=1)
    sample_hr_imgs = valid_hr_imgs[10:19]
    sample_hr_imgs = tl.prepro.threading_data(sample_hr_imgs, fn=crop_sub_imgs_fn, is_random=False)
    tl.vis.save_images(sample_hr_imgs, [ni, ni], save_dir + '/_hr_sample.png')

    sample_lr_imgs = tl.prepro.threading_data(sample_hr_imgs, fn=downsample_fn, down_rate=3) 
    sample_bicubuc_imgs = tl.prepro.threading_data(sample_lr_imgs, fn=upsample_fn, up_rate=3)
    tl.vis.save_images(sample_bicubuc_imgs, [ni, ni], save_dir+'/_bicubic_sample.png')

    single_hr_img = crop_sub_imgs_fn(valid_hr_imgs[1], is_random=False)
    tl.vis.save_image(single_hr_img, save_dir+'/_hr.png')
    single_lr_img = downsample_fn(single_hr_img, down_rate=3)
    single_bicubic_img = upsample_fn(single_lr_img, up_rate=3)
    tl.vis.save_image(single_bicubic_img, save_dir+'/_bicubic.png')

    ###========================== DEFINE MODEL ============================###
    t_image = tf.placeholder('float32', [None, 16, 16, 1], name='input_image')
    net_g = SRGAN_g(t_image, is_train=False, reuse=False)

    ###========================== RESTORE G =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    tl.files.load_and_assign_npz(sess=sess, name=config.MODEL_path, network=net_g)

    ###======================= EVALUATION =============================###
    start_time = time.time()
    single_lr_img = np.expand_dims(single_lr_img, axis=0)
    out = sess.run(net_g.outputs, {t_image: single_lr_img})
    out = np.squeeze(out, axis=0)
    print("took: %4.4fs" % (time.time() - start_time))
    tl.vis.save_image(out, save_dir+'/_srgan.png')

    out1 = sess.run(net_g.outputs, {t_image: sample_lr_imgs})
    tl.vis.save_images(out1, [ni,ni], save_dir+'/_srgan_samples.png')



    # print("LR size: %s /  generated HR size: %s" % (size, out.shape))  # LR size: (339, 510, 3) /  gen HR size: (1, 1356, 2040, 3)
    # print("[*] save images")
    # tl.vis.save_image(out[0], save_dir + '/valid_gen.png')
    # tl.vis.save_image(valid_lr_img, save_dir + '/valid_lr.png')
    # tl.vis.save_image(valid_hr_img, save_dir + '/valid_hr.png')
if __name__ == '__main__':
 	evaluate()