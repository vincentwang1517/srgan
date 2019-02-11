#! /usr/bin/python
# -*- coding: utf8 -*-

import os, time, pickle, random, time, sys
from datetime import datetime
import numpy as np
from time import localtime, strftime
import logging, scipy
import tensorflow as tf
import tensorlayer as tl
from keras.optimizers import SGD
from model import VGG19
from utils import *
from config import config, log_config

###====================== HYPER-PARAMETERS ===========================###
## Adam
batch_size = config.TRAIN.batch_size
lr_init = config.TRAIN.lr_init
beta1 = config.TRAIN.beta1
n_epoch = config.TRAIN.n_epoch
lr_start_decay = config.TRAIN.lr_start_decay
lr_decay_rate = config.TRAIN.decay_rate
decay_every = config.TRAIN.decay_every

ni = int(np.sqrt(9))

def train():
    ## create folders to save result images and trained model
    checkpoint_dir = "checkpoint"  # checkpoint_resize_conv
    tl.files.exists_or_mkdir(checkpoint_dir)

    train_labels, train_hr_imgs = read_csv_data(config.TRAIN.hr_img_path, width=48, height=48, channel=1)
    valid_labels, valid_hr_imgs = read_csv_data(config.VALID.hr_img_path, width=48, height=48, channel=1)

    ###========================== DEFINE MODEL ============================###
    with tf.name_scope("model"):
        ## train inference  ## t = train
        t_image = tf.placeholder('float32', [None, 48, 48, 1], name='input_image')
        t_labels = tf.placeholder(tf.int64, [None], name='input_label')

        net, logits = VGG19(t_image, is_train=True, reuse=False) # net.shape = [None, 7]
        net.print_params(False)
        net.print_layers()

        ## test inference
        net_test, logits_valid = VGG19(t_image, is_train=False, reuse=True)

        # ###========================== DEFINE TRAIN OPS ==========================###
        loss = tl.cost.cross_entropy(output=logits, target=t_labels, name='cross_entropy')
        tf.summary.scalar("loss", loss)

        correct_prediction = tf.equal(tf.argmax(logits_valid, 1), t_labels)
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        tf.summary.scalar("accuracy", accuracy)

        with tf.variable_scope('learning_rate'):
            lr_v = tf.Variable(lr_init, trainable=False)
        
        optimizer = tf.train.AdamOptimizer(lr_v, beta1=beta1).minimize(loss)
        # optimizer = SGD(lr=0.01, decay=1e-6, momentum=0.9).minimize(loss)

        merged = tf.summary.merge_all()

    ###========================== RESTORE MODEL =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)    
    if tl.global_flag['mode'] == 'VGG19_transfer':
        tl.files.load_and_assign_npz(sess=sess, name=config.MODEL.loaded_model, network=net)
        lr_start_decay = 0

    ###========================== SUMMARY =============================###
    train_writer_path = "./log/train"
    tl.files.exists_or_mkdir(train_writer_path)
    train_writer = tf.summary.FileWriter(train_writer_path, graph=tf.get_default_graph())
    test_writer_path = "./log/test"
    tl.files.exists_or_mkdir(test_writer_path)
    test_writer = tf.summary.FileWriter(test_writer_path)

    ###========================= START TRAINING ====================###
    best_valid_accuracy = 0
    sess.run(tf.assign(lr_v, lr_init))
    for epoch in range(0, n_epoch + 1):
        ## update learning rate
        if epoch != 0 and epoch >= lr_start_decay and (epoch % decay_every == 0):
            new_lr_decay = lr_decay_rate**((epoch - lr_start_decay) // decay_every)
            sess.run(tf.assign(lr_v, lr_init * new_lr_decay))
            log = " ** new learning rate: %f " % (lr_init * new_lr_decay)
            print(log)

        epoch_time = time.time()
        total_mse_loss, n_iter, total_acc, valid_iter = 0, 0, 0, 0

        train_hr_imgs, train_labels = shuffle(train_hr_imgs, train_labels) # random shuffle the training data

        for idx in range(0, len(train_hr_imgs), batch_size):
            if idx+batch_size > len(train_hr_imgs):
                break
            
            step_time = time.time()
            # b = batch
            b_imgs = tl.prepro.threading_data(train_hr_imgs[idx:idx + batch_size], fn=crop_sub_imgs_fn, is_random=True)
            b_label = train_labels[idx:idx+batch_size]

            ### ================= BUG POSSIBLE!! =================== ###
            b_imgs = tl.prepro.threading_data(b_imgs, fn=ImageDataGenerator)
            b_imgs = np.expand_dims(b_imgs, axis=-1)
            ### ================= BUG POSSIBLE!! =================== ###

            ## update 
            errM, _, summary = sess.run([loss, optimizer, merged], {t_image: b_imgs, t_labels: b_label})
            train_writer.add_summary(summary, epoch*batch_size+idx)
            sys.stdout.write("Epoch [%2d/%2d] %4d time: %4.4fs, loss: %.8f \r" % (epoch, n_epoch, n_iter, time.time() - step_time, errM))
            sys.stdout.flush()
            total_mse_loss += errM
            n_iter += 1

        # # Validation every epoch
        for idx in range(0, len(valid_hr_imgs), batch_size):
            if idx+batch_size > len(train_hr_imgs):
                break
            step_time = time.time()
            b_imgs = tl.prepro.threading_data(valid_hr_imgs[idx:idx + batch_size], fn=crop_sub_imgs_fn, is_random=True)
            b_label = valid_labels[idx:idx+batch_size]
            errV, accV, summary = sess.run([loss, accuracy, merged], {t_image: b_imgs, t_labels: b_label})
            test_writer.add_summary(summary, epoch*batch_size+idx)
            total_acc += accV
            valid_iter += 1

        log = "\n[*] Epoch: [%2d/%2d] time: %4.4fs, loss: %.8f, acc(validation): %.5f" % (epoch, n_epoch, time.time() - epoch_time, total_mse_loss / n_iter, total_acc / valid_iter)
        print(log)

        ## save model
        if epoch % 10 == 0 :
            tl.files.save_npz(net.all_params, name=checkpoint_dir + '/{}_{}.npz'.format(tl.global_flag['mode'], epoch), sess=sess)

        if total_acc > best_valid_accuracy:
            tl.files.save_npz(net.all_params, name=checkpoint_dir + '/{}_best.npz'.format(tl.global_flag['mode']), sess=sess)


def evaluate():
    ## create folders to save result images
    save_dir = "samples/{}".format(tl.global_flag['mode'])
    tl.files.exists_or_mkdir(save_dir)
    checkpoint_dir = "checkpoint"

    ###====================== PRE-LOAD DATA ===========================###
    # train_hr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.hr_img_path, regx='.*.png', printable=False))
    # train_lr_img_list = sorted(tl.files.load_file_list(path=config.TRAIN.lr_img_path, regx='.*.png', printable=False))
    valid_hr_img_list = sorted(tl.files.load_file_list(path=config.VALID.hr_img_path, regx='.*.png', printable=False))
    valid_lr_img_list = sorted(tl.files.load_file_list(path=config.VALID.lr_img_path, regx='.*.png', printable=False))

    ## If your machine have enough memory, please pre-load the whole train set.
    # train_hr_imgs = tl.vis.read_images(train_hr_img_list, path=config.TRAIN.hr_img_path, n_threads=32)
    # for im in train_hr_imgs:
    #     print(im.shape)
    valid_lr_imgs = tl.vis.read_images(valid_lr_img_list, path=config.VALID.lr_img_path, n_threads=32)
    # for im in valid_lr_imgs:
    #     print(im.shape)
    valid_hr_imgs = tl.vis.read_images(valid_hr_img_list, path=config.VALID.hr_img_path, n_threads=32)
    # for im in valid_hr_imgs:
    #     print(im.shape)

    ###========================== DEFINE MODEL ============================###
    imid = 64  # 0: 企鹅  81: 蝴蝶 53: 鸟  64: 古堡
    valid_lr_img = valid_lr_imgs[imid]
    valid_hr_img = valid_hr_imgs[imid]
    # valid_lr_img = get_imgs_fn('test.png', 'data2017/')  # if you want to test your own image
    valid_lr_img = (valid_lr_img / 127.5) - 1  # rescale to ［－1, 1]
    # print(valid_lr_img.min(), valid_lr_img.max())

    size = valid_lr_img.shape
    # t_image = tf.placeholder('float32', [None, size[0], size[1], size[2]], name='input_image') # the old version of TL need to specify the image size
    t_image = tf.placeholder('float32', [1, None, None, 3], name='input_image')

    net_g = SRGAN_g(t_image, is_train=False, reuse=False)

    ###========================== RESTORE G =============================###
    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=False))
    tl.layers.initialize_global_variables(sess)
    tl.files.load_and_assign_npz(sess=sess, name=checkpoint_dir + '/g_srgan.npz', network=net_g)

    ###======================= EVALUATION =============================###
    start_time = time.time()
    out = sess.run(net_g.outputs, {t_image: [valid_lr_img]})
    print("took: %4.4fs" % (time.time() - start_time))

    print("LR size: %s /  generated HR size: %s" % (size, out.shape))  # LR size: (339, 510, 3) /  gen HR size: (1, 1356, 2040, 3)
    print("[*] save images")
    tl.vis.save_image(out[0], save_dir + '/valid_gen.png')
    tl.vis.save_image(valid_lr_img, save_dir + '/valid_lr.png')
    tl.vis.save_image(valid_hr_img, save_dir + '/valid_hr.png')

    out_bicu = scipy.misc.imresize(valid_lr_img, [size[0] * 4, size[1] * 4], interp='bicubic', mode=None)
    tl.vis.save_image(out_bicu, save_dir + '/valid_bicubic.png')


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='VGG_vin', help='VGG_vin, evaluate, VGG19_transfer')
    ''' 
    VGG19_trainsfer: continuously train a model from a pretrained VGG19 network (see config.MODEL.loaded_model)
    '''

    args = parser.parse_args()

    tl.global_flag['mode'] = args.mode

    if tl.global_flag['mode'] == 'VGG_vin':
        train()
    elif tl.global_flag['mode'] == 'VGG19_transfer':
        lr_init = lr_init * 0.1
        train()
    elif tl.global_flag['mode'] == 'evaluate':
        evaluate()
    else:
        raise Exception("Unknow --mode")
