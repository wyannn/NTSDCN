# -*- coding: utf-8 -*-
"""
Created on Thu Jun  7 14:54:09 2018

@author: student
"""
from utils import read_data, laplacian_loss
import time
import os
import tensorflow as tf
import numpy as np

class NTSDCN(object):

    def __init__(self, sess, image_size = 64, label_size = 64, batch_size = 64,
                 c_dim = 1, checkpoint_dir = None, training = True):

        self.sess = sess
        self.is_grayscale = (c_dim == 1)
        self.image_size = image_size
        self.label_size = label_size
        self.batch_size = batch_size
        self.c_dim = c_dim
        self.checkpoint_dir = checkpoint_dir
        self.training = training
        
        self.build_model()

    def build_model(self):
        
        self.images_mos = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, self.c_dim], name='images_g')
        self.images_r = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, self.c_dim], name='images_r')
        self.images_b = tf.placeholder(tf.float32, [self.batch_size, self.image_size, self.image_size, self.c_dim], name='images_b')
        self.labels_g = tf.placeholder(tf.float32, [self.batch_size, self.label_size, self.label_size, self.c_dim], name='labels_g')   
        self.labels_r = tf.placeholder(tf.float32, [self.batch_size, self.label_size, self.label_size, self.c_dim], name='labels_r')   
        self.labels_b = tf.placeholder(tf.float32, [self.batch_size, self.label_size, self.label_size, self.c_dim], name='labels_b')

        self.labels_rgb = tf.concat([self.labels_r, self.labels_g, self.labels_b], 3)
        
        self.lp_g, self.pred_g = self.model_g()
        self.lp_r, self.pred_r = self.model_r(self.pred_g)
        self.lp_b, self.pred_b = self.model_b(self.pred_g)
        
        self.pred_rgb = tf.concat([self.pred_r, self.pred_g, self.pred_b], 3)
                
        self.loss_g = tf.reduce_mean(tf.losses.absolute_difference(self.labels_g, self.pred_g)) + 0.01 * self.lp_g
        self.loss_r = tf.reduce_mean(tf.losses.absolute_difference(self.labels_r, self.pred_r)) + 0.01 * self.lp_r
        self.loss_b = tf.reduce_mean(tf.losses.absolute_difference(self.labels_b, self.pred_b)) + 0.01 * self.lp_b
        self.loss_rgb = tf.reduce_mean(tf.losses.absolute_difference(self.labels_rgb, self.pred_rgb))

        self.saver = tf.train.Saver()

    def train(self, Config):
        
        data_dir = os.path.join('./{}'.format(Config.checkpoint_dir), Config.data_dir) 
        
        train_data_mos, train_data_r, train_data_b, train_label_r, train_label_g, train_label_b = read_data(data_dir, Config)          

        self.train_g = tf.train.AdamOptimizer(Config.learning_rate).minimize(self.loss_g, var_list=self.vars_g)
        self.train_r = tf.train.AdamOptimizer(Config.learning_rate).minimize(self.loss_r, var_list=self.vars_r)
        self.train_b = tf.train.AdamOptimizer(Config.learning_rate).minimize(self.loss_b, var_list=self.vars_b)
        self.train_rgb = tf.train.AdamOptimizer(Config.learning_rate).minimize(self.loss_rgb, var_list=[self.vars_r, self.vars_g, self.vars_b])

        tf.global_variables_initializer().run()
                
        counter = 0
        start_time = time.time()

        if self.load(self.checkpoint_dir):
            print("Load SUCCESS.")
        else:
            print("Load failed!")

        print("Training...")

        for ep in range(Config.epoch):
            batch_idxs = len(train_data_mos) // Config.batch_size
            
            permutation = np.random.permutation(train_data_mos.shape[0])

            minn = 10000
            for idx in range(0, batch_idxs):
                batch_images_mos = train_data_mos[permutation[idx*Config.batch_size : (idx+1)*Config.batch_size]]
                batch_images_r = train_data_r[permutation[idx*Config.batch_size : (idx+1)*Config.batch_size]]
                batch_images_b = train_data_b[permutation[idx*Config.batch_size : (idx+1)*Config.batch_size]]
                batch_labels_g = train_label_g[permutation[idx*Config.batch_size : (idx+1)*Config.batch_size]]
                batch_labels_r = train_label_r[permutation[idx*Config.batch_size : (idx+1)*Config.batch_size]]
                batch_labels_b = train_label_b[permutation[idx*Config.batch_size : (idx+1)*Config.batch_size]]
                
                counter += 1
                
                if ep < Config.epoch // 4 * 1:
                    _, err = self.sess.run([self.train_g, self.loss_g], feed_dict={self.images_mos: batch_images_mos, self.labels_g: batch_labels_g})
                elif ep < Config.epoch // 4 * 2:
                    _, err = self.sess.run([self.train_r, self.loss_r], feed_dict={self.images_mos: batch_images_mos, self.images_r: batch_images_r, self.labels_r: batch_labels_r, self.labels_g: batch_labels_g})
                elif ep < Config.epoch // 4 * 3:
                    _, err = self.sess.run([self.train_b, self.loss_b], feed_dict={self.images_mos: batch_images_mos, self.images_b: batch_images_b, self.labels_b: batch_labels_b, self.labels_g: batch_labels_g})
                else:
                    _, err = self.sess.run([self.train_rgb, self.loss_rgb], feed_dict={self.images_mos: batch_images_mos, self.images_r: batch_images_r, self.images_b: batch_images_b, self.labels_r: batch_labels_r, self.labels_g: batch_labels_g, self.labels_b: batch_labels_b})
                
 
                if counter % 100 == 0:   
                    print("Epoch: [%2d], step: [%2d], time: [%4.4f], loss: [%.8f]" \
                             % ((ep+1), counter, time.time()-start_time, err))
                
                if counter % 10000 == 0:
                   self.save(Config.checkpoint_dir, counter)
                if err <= minn:
                    minn = err
                    self.save(Config.checkpoint_dir, counter)
            self.save(Config.checkpoint_dir, counter)
                    
    
    def model_g(self):
        
        with tf.variable_scope('g'):
        
            g_conv_low1 = tf.contrib.layers.conv2d(self.images_mos, 32, kernel_size=(3,3), stride=1, padding='SAME')
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
            g_out = g_residual + self.images_mos
            
            g_lp = (laplacian_loss(g_b1_conv1_add) + laplacian_loss(g_b2_conv1_add) + laplacian_loss(g_b3_conv1_add)) / 3
            
            self.vars_g = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'g')
            
            return g_lp, g_out
      
    def model_r(self, g):
        
        with tf.variable_scope('r'):
                        
            r_b1_alpha = tf.contrib.layers.conv2d(g, 32, kernel_size=(3,3), stride=1, padding='SAME', activation_fn=tf.sigmoid)
            r_b1_beta = tf.contrib.layers.conv2d(g, 32, kernel_size=(3,3), stride=1, padding='SAME', activation_fn=None)
            
            r_b1_con = tf.concat([self.images_mos, self.images_r], 3)
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
            r_out = r_b2_residual + self.images_r
            
            r_lp = laplacian_loss(r_b1_conv1_add)
            
            self.vars_r = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'r')
            
            return r_lp, r_out
        
    def model_b(self, g):
        
        with tf.variable_scope('b'):
                        
            b_b1_alpha = tf.contrib.layers.conv2d(g, 32, kernel_size=(3,3), stride=1, padding='SAME', activation_fn=tf.sigmoid)
            b_b1_beta = tf.contrib.layers.conv2d(g, 32, kernel_size=(3,3), stride=1, padding='SAME', activation_fn=None)
            
            b_b1_con = tf.concat([self.images_mos, self.images_b], 3)
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
            b_out = b_b2_residual + self.images_b
            
            b_lp = laplacian_loss(b_b1_conv1_add)
            
            self.vars_b = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'b')
            
            return b_lp, b_out
        

    def save(self, checkpoint_dir, step):
        model_name = "TRY.model"
        model_dir = "%s_%s" % ("try", self.label_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        if not os.path.exists(checkpoint_dir):
            os.makedirs(checkpoint_dir)

        self.saver.save(self.sess, os.path.join(checkpoint_dir, model_name),
                        global_step=step)

    def load(self, checkpoint_dir):
        print("Reading checkpoints...")
        model_dir = "%s_%s" % ("try", self.label_size)
        checkpoint_dir = os.path.join(checkpoint_dir, model_dir)

        ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
        if ckpt and ckpt.model_checkpoint_path:
            ckpt_name = os.path.basename(ckpt.model_checkpoint_path)
            self.saver.restore(self.sess, os.path.join(checkpoint_dir, ckpt_name))
            return True
        else:
            return False