# -*- coding: utf-8 -*-
"""
Created on Fri Jan 19 14:18:35 2018

@author: student

"""

import tensorflow as tf
import os
import numpy as np
from PIL import Image
from utils import laplacian_loss

test_mos = '../test_kodak/mos'
pic_list_mos = os.listdir(test_mos)

test_r = '../test_kodak/0r'
pic_list_r = os.listdir(test_r)

test_b = '../test_kodak/0b'
pic_list_b = os.listdir(test_b)

checkpoint_dir = 'checkpoint/try_64'
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
model_list = ckpt.all_model_checkpoint_paths

for ckpt_name in model_list:
    model_dir = ckpt_name
    res_dir = '../res/' + ckpt_name[-8:]
    os.mkdir(res_dir)
    
    for i in range(len(pic_list_mos)):
        if ('m' in test_mos and os.path.splitext(pic_list_mos[i])[1] == '.bmp'):
            
            image_mos = np.array(Image.open(test_mos + '/' + pic_list_mos[i])).astype(np.float32) / 255.
            image_r = np.array(Image.open(test_r + '/' + pic_list_r[i])).astype(np.float32) / 255.
            image_b = np.array(Image.open(test_b + '/' + pic_list_b[i])).astype(np.float32) / 255.
            
            input_image_mos = np.reshape(image_mos, [1, image_mos.shape[0], image_mos.shape[1], 1])
            input_image_r = np.reshape(image_r, [1, image_r.shape[0], image_r.shape[1], 1])
            input_image_b = np.reshape(image_b, [1, image_b.shape[0], image_b.shape[1], 1])
            
            tf.reset_default_graph()
        
            images_mos = tf.placeholder(tf.float32, [1, image_mos.shape[0], image_mos.shape[1], 1], name='images_mos')
            images_r = tf.placeholder(tf.float32, [1, image_r.shape[0], image_r.shape[1], 1], name='images_r')
            images_b = tf.placeholder(tf.float32, [1, image_b.shape[0], image_b.shape[1], 1], name='images_b')
            
            def model_g():
        
                with tf.variable_scope('g'):
                
                    g_conv_low1 = tf.contrib.layers.conv2d(images_mos, 32, kernel_size=(3,3), stride=1, padding='SAME')
                    g_conv_low2 = tf.contrib.layers.conv2d(g_conv_low1, 64, kernel_size=(3,3), stride=1, padding='SAME', activation_fn=None)
                    
                    g_b1_conv1_1 = tf.contrib.layers.conv2d(g_conv_low2, 64, kernel_size=(3,3), stride=1, padding='SAME')
                    g_b1_conv1_2 = tf.contrib.layers.conv2d(g_b1_conv1_1, 64, kernel_size=(3,3), stride=1, padding='SAME', rate=2, biases_initializer=None)
                    g_b1_conv1_3 = tf.contrib.layers.conv2d(g_b1_conv1_2, 64, kernel_size=(3,3), stride=1, padding='SAME', rate=2, biases_initializer=None)
                    g_b1_conv1_4 = tf.contrib.layers.conv2d(g_b1_conv1_3, 64, kernel_size=(3,3), stride=1, padding='SAME', activation_fn=None)
                    g_b1_conv1_add = g_conv_low2 - 0.1 * g_b1_conv1_4
                    
                    g_b1_conv_5 = tf.contrib.layers.conv2d(g_b1_conv1_add, 64, kernel_size=(3,3), stride=1, padding='SAME')  
                    g_b1_conv_6 = tf.contrib.layers.conv2d(g_b1_conv_5, 64, kernel_size=(3,3), stride=1, padding='SAME', activation_fn=None) 
                    g_b1_conv_add2 = g_b1_conv_6 + 0.1 * g_b1_conv1_add
                    
                    g_b2_conv1_1 = tf.contrib.layers.conv2d(g_b1_conv_add2, 64, kernel_size=(3,3), stride=1, padding='SAME')
                    g_b2_conv1_2 = tf.contrib.layers.conv2d(g_b2_conv1_1, 64, kernel_size=(3,3), stride=1, padding='SAME', rate=2, biases_initializer=None)
                    g_b2_conv1_3 = tf.contrib.layers.conv2d(g_b2_conv1_2, 64, kernel_size=(3,3), stride=1, padding='SAME', rate=2, biases_initializer=None)
                    g_b2_conv1_4 = tf.contrib.layers.conv2d(g_b2_conv1_3, 64, kernel_size=(3,3), stride=1, padding='SAME', activation_fn=None)
                    g_b2_conv1_add = g_b1_conv_add2 - 0.1 * g_b2_conv1_4
                    
                    g_b2_conv_5 = tf.contrib.layers.conv2d(g_b2_conv1_add, 64, kernel_size=(3,3), stride=1, padding='SAME')  
                    g_b2_conv_6 = tf.contrib.layers.conv2d(g_b2_conv_5, 64, kernel_size=(3,3), stride=1, padding='SAME', activation_fn=None) 
                    g_b2_conv_add2 = g_b2_conv_6 + 0.1 * g_b2_conv1_add
                    
                    g_b3_conv1_1 = tf.contrib.layers.conv2d(g_b2_conv_add2, 64, kernel_size=(3,3), stride=1, padding='SAME')
                    g_b3_conv1_2 = tf.contrib.layers.conv2d(g_b3_conv1_1, 64, kernel_size=(3,3), stride=1, padding='SAME', rate=2, biases_initializer=None)
                    g_b3_conv1_3 = tf.contrib.layers.conv2d(g_b3_conv1_2, 64, kernel_size=(3,3), stride=1, padding='SAME', rate=2, biases_initializer=None)
                    g_b3_conv1_4 = tf.contrib.layers.conv2d(g_b3_conv1_3, 64, kernel_size=(3,3), stride=1, padding='SAME', activation_fn=None)
                    g_b3_conv1_add = g_b2_conv_add2 - 0.1 * g_b3_conv1_4
                    
                    g_b3_conv_5 = tf.contrib.layers.conv2d(g_b3_conv1_add, 64, kernel_size=(3,3), stride=1, padding='SAME')  
                    g_b3_conv_6 = tf.contrib.layers.conv2d(g_b3_conv_5, 64, kernel_size=(3,3), stride=1, padding='SAME', activation_fn=None) 
                    g_b3_conv_add2 = g_b3_conv_6 + 0.1 * g_b3_conv1_add
                    
                    g_b3_total = tf.concat([g_b1_conv_add2, g_b2_conv_add2, g_b3_conv_add2], 3)
                    g_b3_red = tf.contrib.layers.conv2d(g_b3_total, 64, kernel_size=(1,1), stride=1, padding='SAME')
                    
                    g_conv_high1 = tf.contrib.layers.conv2d(g_b3_red, 64, kernel_size=(3,3), stride=1, padding='SAME')
                    g_conv_high2 = tf.contrib.layers.conv2d(g_conv_high1, 64, kernel_size=(3,3), stride=1, padding='SAME')
                    g_residual = tf.contrib.layers.conv2d(g_conv_high2, 1, kernel_size=(3,3), stride=1, padding='SAME', activation_fn=None)
                    g_out = g_residual + images_mos
                    
                    g_lp = (laplacian_loss(g_b1_conv1_add) + laplacian_loss(g_b2_conv1_add) + laplacian_loss(g_b3_conv1_add)) / 3
                                        
                    return g_lp, g_out
              
            def model_r(g):
        
                with tf.variable_scope('r'):
                                
                    r_b1_alpha = tf.contrib.layers.conv2d(g, 32, kernel_size=(3,3), stride=1, padding='SAME', activation_fn=tf.sigmoid)
                    r_b1_beta = tf.contrib.layers.conv2d(g, 32, kernel_size=(3,3), stride=1, padding='SAME', activation_fn=None)
                    
                    r_b1_con = tf.concat([images_mos, images_r], 3)
                    r_b1_conv_low1 = tf.contrib.layers.conv2d(r_b1_con, 32, kernel_size=(3,3), stride=1, padding='SAME', activation_fn=None)
                    r_b1_transform =  r_b1_alpha * r_b1_conv_low1 + r_b1_beta
                    r_b1_conv_low2 = tf.contrib.layers.conv2d(r_b1_transform, 64, kernel_size=(3,3), stride=1, padding='SAME')
                    
                    r_b1_conv1_1 = tf.contrib.layers.conv2d(r_b1_conv_low2, 64, kernel_size=(3,3), stride=1, padding='SAME')
                    r_b1_conv1_2 = tf.contrib.layers.conv2d(r_b1_conv1_1, 64, kernel_size=(3,3), stride=1, padding='SAME', rate=2, biases_initializer=None)
                    r_b1_conv1_3 = tf.contrib.layers.conv2d(r_b1_conv1_2, 64, kernel_size=(3,3), stride=1, padding='SAME', rate=2, biases_initializer=None)
                    r_b1_conv1_5 = tf.contrib.layers.conv2d(r_b1_conv1_3, 64, kernel_size=(3,3), stride=1, padding='SAME', activation_fn=None)
                    r_b1_conv1_add = r_b1_conv_low2 - r_b1_conv1_5
                    
                    r_b2_conv_high1 = tf.contrib.layers.conv2d(r_b1_conv1_add, 64, kernel_size=(3,3), stride=1, padding='SAME')  
                    r_b2_conv_high2 = tf.contrib.layers.conv2d(r_b2_conv_high1, 64, kernel_size=(3,3), stride=1, padding='SAME', activation_fn=None) 
                    r_b2_conv_add = r_b2_conv_high2 + r_b1_conv1_add
                    r_b2_residual = tf.contrib.layers.conv2d(r_b2_conv_add, 1, kernel_size=(3,3), stride=1, padding='SAME', activation_fn=None)            
                    r_out = r_b2_residual + images_r
                    
                    r_lp = laplacian_loss(r_b1_conv1_add)
                                        
                    return r_lp, r_out
                
            def model_b(g):
                
                with tf.variable_scope('b'):
                                
                    b_b1_alpha = tf.contrib.layers.conv2d(g, 32, kernel_size=(3,3), stride=1, padding='SAME', activation_fn=tf.sigmoid)
                    b_b1_beta = tf.contrib.layers.conv2d(g, 32, kernel_size=(3,3), stride=1, padding='SAME', activation_fn=None)
                    
                    b_b1_con = tf.concat([images_mos, images_b], 3)
                    b_b1_conv_low1 = tf.contrib.layers.conv2d(b_b1_con, 32, kernel_size=(3,3), stride=1, padding='SAME', activation_fn=None)
                    b_b1_transform =  b_b1_alpha * b_b1_conv_low1 + b_b1_beta
                    b_b1_conv_low2 = tf.contrib.layers.conv2d(b_b1_transform, 64, kernel_size=(3,3), stride=1, padding='SAME')
                    
                    b_b1_conv1_1 = tf.contrib.layers.conv2d(b_b1_conv_low2, 64, kernel_size=(3,3), stride=1, padding='SAME')
                    b_b1_conv1_2 = tf.contrib.layers.conv2d(b_b1_conv1_1, 64, kernel_size=(3,3), stride=1, padding='SAME', rate=2, biases_initializer=None)
                    b_b1_conv1_3 = tf.contrib.layers.conv2d(b_b1_conv1_2, 64, kernel_size=(3,3), stride=1, padding='SAME', rate=2, biases_initializer=None)
                    b_b1_conv1_5 = tf.contrib.layers.conv2d(b_b1_conv1_3, 64, kernel_size=(3,3), stride=1, padding='SAME', activation_fn=None)
                    b_b1_conv1_add = b_b1_conv_low2 - b_b1_conv1_5
                    
                    b_b2_conv_high1 = tf.contrib.layers.conv2d(b_b1_conv1_add, 64, kernel_size=(3,3), stride=1, padding='SAME')  
                    b_b2_conv_high2 = tf.contrib.layers.conv2d(b_b2_conv_high1, 64, kernel_size=(3,3), stride=1, padding='SAME', activation_fn=None)
                    b_b2_conv_add = b_b2_conv_high2 + b_b1_conv1_add
                    b_b2_residual = tf.contrib.layers.conv2d(b_b2_conv_add, 1, kernel_size=(3,3), stride=1, padding='SAME', activation_fn=None)            
                    b_out = b_b2_residual + images_b
                    
                    b_lp = laplacian_loss(b_b1_conv1_add)
                                        
                    return b_lp, b_out
                                 
            x, g = model_g()
            y, r = model_r(g)
            z, b = model_b(g)
            rgb = tf.concat([r, g, b], 3)
            
            output = tf.clip_by_value(rgb, 0, 1)
            
            saver = tf.train.Saver() 
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.4)
            with tf.Session(config=tf.ConfigProto(gpu_options=gpu_options)) as sess:
                saver.restore(sess, model_dir)
                results = sess.run(output, feed_dict={images_mos: input_image_mos, images_r: input_image_r, images_b: input_image_b})
                
            res = (np.squeeze(results)) * 255.
            Image.fromarray(np.uint8(res)).save(res_dir + '/' + pic_list_mos[i][:-4] + '.bmp')
