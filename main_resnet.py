#! /usr/bin/python
# -*- coding: utf8 -*-

import os, time, pickle, random, time, sys
from datetime import datetime
import numpy as np
from time import localtime, strftime
import logging, scipy
import tensorflow as tf
import tensorlayer as tl
import tensorlayer as tl
from tensorlayer.layers import *
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


def train():
    ## create folders to save result images and trained model
    save_dir_ginit = "resnet/samples/{}_ginit".format(tl.global_flag['mode'])
    save_dir_gan = "resnet/samples/{}_train".format(tl.global_flag['mode'])
    save_dir_valid = "resnet/samples/{}_valid".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir_ginit)
    tl.files.exists_or_mkdir(save_dir_gan)
    tl.files.exists_or_mkdir(save_dir_valid)
    checkpoint_dir = "resnet/checkpoint"  # checkpoint_resize_conv
    tl.files.exists_or_mkdir(checkpoint_dir)

    train_hr_imgs = read_csv_data(config.TRAIN.hr_img_path, width=48, height=48, channel=1)
    valid_hr_imgs = read_csv_data(config.VALID.hr_img_path, width=48, height=48, channel=1)

    ###========================== DEFINE MODEL ============================###
    ## train inference  ## t = train
    t_image = tf.placeholder('float32', [None, 16, 16, 1], name='t_image_input_to_RESNET_generator')
    t_target_image = tf.placeholder('float32', [None, 48, 48, 1], name='t_target_image')

    net_g = Resnet1(t_image, is_train=True, reuse=False, nb_block=16)

    net_g.print_params(False)
    net_g.print_layers()

    net_g_test = Resnet1(t_image, is_train=False, reuse=True)

    # ###========================== DEFINE TRAIN OPS ==========================###
    # g_loss: for generator
    mse_loss = tl.cost.mean_squared_error(net_g.outputs, t_target_image, is_mean=True)
    
    g_loss = mse_loss   

    g_vars = tl.layers.get_variables_with_name('Resnet1', True, True)

    with tf.variable_scope('learning_rate'):
        lr_v = tf.Variable(lr_init, trainable=False)

    ## Resnet
    g_optim = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(g_loss, var_list=g_vars)

    # ###========================== RESTORE MODEL =============================###
    # sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    # tl.layers.initialize_global_variables(sess)
    # if tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_{}.npz'.format(tl.global_flag['mode']), network=net_g) is False:
    #     tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_{}_init.npz'.format(tl.global_flag['mode']), network=net_g)
    # tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/d_{}.npz'.format(tl.global_flag['mode']), network=net_d)

    ###============================= TRAINING ===============================###
    ## use first `batch_size` of train set to have a quick test during training
    sample_imgs = train_hr_imgs[0:9]
    valid_imgs = valid_hr_imgs[44:53]
    sample_imgs_384 = tl.prepro.threading_data(sample_imgs, fn=crop_sub_imgs_fn, is_random=False)
    valid_imgs_48 = tl.prepro.threading_data(valid_imgs, fn=crop_sub_imgs_fn, is_random=False)
    print('sample HR sub-image:', sample_imgs_384.shape, sample_imgs_384.min(), sample_imgs_384.max())
    sample_imgs_96 = tl.prepro.threading_data(sample_imgs_384, fn=downsample_fn, down_rate=3)
    valid_imgs_16 = tl.prepro.threading_data(valid_imgs_48, fn=downsample_fn, down_rate=3)
    print('sample LR sub-image:', sample_imgs_96.shape, sample_imgs_96.min(), sample_imgs_96.max())
    tl.vis.save_images(sample_imgs_96, [ni, ni], save_dir_ginit + '/_train_sample_16.png')
    tl.vis.save_images(sample_imgs_384, [ni, ni], save_dir_ginit + '/_train_sample_48.png')
    tl.vis.save_images(sample_imgs_96, [ni, ni], save_dir_gan + '/_train_sample_16.png')
    tl.vis.save_images(sample_imgs_384, [ni, ni], save_dir_gan + '/_train_sample_48.png')
    tl.vis.save_images(valid_imgs_48, [ni, ni], save_dir_valid + '/_valid_sample_48.png')
    tl.vis.save_images(valid_imgs_16, [ni, ni], save_dir_valid + '/_valid_sample_16.png')
    sample_hr_imgs_bicubic = tl.prepro.threading_data(sample_imgs_96, fn=upsample_fn, up_rate=3)
    valid_hr_imgs_bicubic = tl.prepro.threading_data(valid_imgs_16, fn=upsample_fn, up_rate=3)
    tl.vis.save_images(sample_hr_imgs_bicubic, [ni, ni], save_dir_ginit + '/_sample_bicubic_48.png')
    tl.vis.save_images(valid_hr_imgs_bicubic, [ni,ni], save_dir_valid + '/_valid_sample_bicubic_48.png')


    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    ###========================= initialize G ====================###
    ## fixed learning rate
    sess.run(tf.assign(lr_v, lr_init))
    train_writer_path = "./resnet/log/train"
    tl.files.exists_or_mkdir(train_writer_path)
    train_writer = tf.summary.FileWriter(train_writer_path, graph=tf.get_default_graph())
    print(" ** fixed learning rate: %f (for init G)" % lr_init)

    ###========================= train GAN (SRGAN) =========================###
    for epoch in range(0, n_epoch + 1):
        ## update learning rate
        if epoch != 0 and (epoch % decay_every == 0):
            new_lr_decay = lr_decay**(epoch // decay_every)
            sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
            log = " ** new learning rate: %f (for GAN)" % (lr_init * new_lr_decay)
            print(log)
        elif epoch == 0:
            sess.run(tf.assign(lr_v, lr_init))
            log = " ** init lr: %f  decay_every_init: %d, lr_decay: %f (for GAN)" % (lr_init, decay_every, lr_decay)
            print(log)

        epoch_time = time.time()
        total_d_loss, total_g_loss, n_iter = 0, 0, 0

        for idx in range(0, len(train_hr_imgs), batch_size):
            step_time = time.time()
            b_imgs_384 = tl.prepro.threading_data(train_hr_imgs[idx:idx + batch_size], fn=crop_sub_imgs_fn, is_random=True)
            b_imgs_96 = tl.prepro.threading_data(b_imgs_384, fn=downsample_fn)

            ## update G
            errM, _ = sess.run([mse_loss, g_optim], {t_image: b_imgs_96, t_target_image: b_imgs_384})

            sys.stdout.write("Epoch [%2d/%2d] %4d time: %4.4fs, mse_loss: %.8f \r" %
                  (epoch, n_epoch, n_iter, time.time() - step_time, errM))
            sys.stdout.flush()
            total_g_loss += errM
            n_iter += 1

        log = "[*] Epoch: [%2d/%2d] time: %4.4fs, g_loss: %.8f" % (epoch, n_epoch, time.time() - epoch_time, total_g_loss / n_iter)
        print(log)

        ## quick evaluation on train set
        if (epoch != 0) and (epoch % 10 == 0):
            out = sess.run(net_g_test.outputs, {t_image: sample_imgs_96})  #; print('gen sub-image:', out.shape, out.min(), out.max())
            print("[*] save images")
            tl.vis.save_images(out, [ni, ni], save_dir_gan + '/train_%d.png' % epoch)

        ## quick evaluation on validation set
        if (epoch != 0) and (epoch % 10 == 0):
            out = sess.run(net_g_test.outputs, {t_image: valid_imgs_16})  #; print('gen sub-image:', out.shape, out.min(), out.max())
            tl.vis.save_images(out, [ni, ni], save_dir_valid + '/valid_gan_%d.png' % epoch)

        ## save model
        if epoch % 10 == 0:
            tl.files.save_npz(net_g.all_params, name=checkpoint_dir + '/{}_{}.npz'.format(tl.global_flag['mode'], epoch), sess=sess)


def Resnet1(t_image, is_train=False, reuse=False, nb_block=16):
    w_init = tf.random_normal_initializer(stddev=0.02)
    b_init = None  # tf.constant_initializer(value=0.0)
    g_init = tf.random_normal_initializer(1., 0.02)
    with tf.variable_scope("Resnet1", reuse=reuse) as vs:
        # tl.layers.set_name_reuse(reuse) # remove for TL 1.8.0+
        n = InputLayer(t_image, name='in')
        n = Conv2d(n, 64, (3, 3), (1, 1), act=tf.nn.relu, padding='SAME', W_init=w_init, name='n64s1/c')
        temp = n

        # first B residual blocks (16x16)
        for i in range(8):
            nn = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/%s/c1' % i)
            nn = BatchNormLayer(nn, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='n64s1/%s/b1' % i)
            nn = Conv2d(nn, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/%s/c2' % i)
            nn = BatchNormLayer(nn, is_train=is_train, gamma_init=g_init, name='n64s1/%s/b2' % i)
            nn = ElementwiseLayer([n, nn], tf.add, name='b_residual_add/%s' % i)
            n = nn

        n = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s1/c/m')
        n = BatchNormLayer(n, is_train=is_train, gamma_init=g_init, name='n64s1/b/m')
        n = ElementwiseLayer([n, temp], tf.add, name='add3')
        # first B residual blacks end

        n = Conv2d(n, 576, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n256s1/1')
        n = SubpixelConv2d(n, scale=3, n_out_channel=None, act=tf.nn.relu, name='pixelshufflerx2/1')
        # image size: (64x64)

        for i in range(8):
            nn = Conv2d(n, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s2/%s/c1' % i)
            nn = BatchNormLayer(nn, act=tf.nn.relu, is_train=is_train, gamma_init=g_init, name='n64s2/%s/b1' % i)
            nn = Conv2d(nn, 64, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, b_init=b_init, name='n64s2/%s/c2' % i)
            nn = BatchNormLayer(nn, is_train=is_train, gamma_init=g_init, name='n64s2/%s/b2' % i)
            nn = ElementwiseLayer([n, nn], tf.add, name='b_residual_add2/%s' % i)
            n = nn

        # n = Conv2d(n, 256, (3, 3), (1, 1), act=None, padding='SAME', W_init=w_init, name='n256s1/2')
        # n = SubpixelConv2d(n, scale=2, n_out_channel=None, act=tf.nn.relu, name='pixelshufflerx2/2')

        n = Conv2d(n, 1, (1, 1), (1, 1), act=tf.nn.tanh, padding='SAME', W_init=w_init, name='out')
        return n

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='resnet', help='resnet')

    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode

    if tl.global_flag['mode'] == 'resnet':
        train()
    else:
        raise Exception("Unknow --mode")
