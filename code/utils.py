# -*- coding: utf-8 -*-
"""
Created on Wed Nov 29 22:12:04 2017

@author: wang yan
"""
import h5py
import numpy as np
import tensorflow as tf
    
def read_data(path, Config):

    with h5py.File(path, 'r') as hf:
        data_mos = np.array(hf.get('data_mos'))
        data_r = np.array(hf.get('data_r'))
        data_b = np.array(hf.get('data_b'))
        label_g = np.array(hf.get('label_g'))
        label_r = np.array(hf.get('label_r'))
        label_b = np.array(hf.get('label_b'))
        data_mos = np.reshape(data_mos, [data_mos.shape[0], Config.image_size, Config.image_size, Config.c_dim])
        data_r = np.reshape(data_r, [data_r.shape[0], Config.image_size, Config.image_size, Config.c_dim])
        data_b = np.reshape(data_b, [data_b.shape[0], Config.image_size, Config.image_size, Config.c_dim])
        label_g = np.reshape(label_g, [label_g.shape[0], Config.label_size, Config.label_size, Config.c_dim])
        label_r = np.reshape(label_r, [label_r.shape[0], Config.label_size, Config.label_size, Config.c_dim])
        label_b = np.reshape(label_b, [label_b.shape[0], Config.label_size, Config.label_size, Config.c_dim])
        return data_mos, data_r, data_b, label_r, label_g, label_b


def laplacian_loss(img):
    imsize = img.get_shape().as_list()
    mask = tf.constant([[-1, -1, -1],
                        [-1, 8., -1], 
                        [-1, -1, -1]])  
    img = tf.reshape(img, [imsize[0] * imsize[-1], imsize[1], imsize[2], 1])
    mask = tf.reshape(mask, [3, 3, 1, 1])
    energy = tf.nn.conv2d(img, mask, strides=[1, 1, 1, 1], padding='VALID')
    loss = tf.reduce_mean(tf.abs(energy))
    
    return loss
        