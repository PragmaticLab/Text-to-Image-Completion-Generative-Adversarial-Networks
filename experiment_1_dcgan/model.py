import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import tensorflow as tf 
from ops import *


def conv_out_size_same(size, stride):
    return int(math.ceil(float(size) / float(stride)))

class DCGAN(object):

    def __init__(self, input_height=108, input_width=108, 
        batch_size=64, output_height=64, output_width=64,
        z_dim=100, gf_dim=64, df_dim=64, c_dim=3, lr=0.0001):

        self.batch_size = batch_size
        self.lr = lr

        self.input_height = input_height
        self.input_width = input_width
        self.output_height = output_height
        self.output_width = output_width

        self.z_dim = z_dim
        self.c_dim = c_dim

        self.gf_dim = gf_dim
        self.df_dim = df_dim

    def generator(self, z, reuse=False):
        '''
        z - (?, 100)
        h0 - (?, 4, 4, 512)
        h1 - (64, 8, 8, 256)
        h2 - (64, 16, 16, 128)
        h3 - (64, 32, 32, 64)
        h4 - (64, 64, 64, 3)
        '''
        with tf.variable_scope("generator", reuse=reuse):
            s_h, s_w = self.output_height, self.output_width
            s_h2, s_w2 = conv_out_size_same(s_h, 2), conv_out_size_same(s_w, 2)
            s_h4, s_w4 = conv_out_size_same(s_h2, 2), conv_out_size_same(s_w2, 2)
            s_h8, s_w8 = conv_out_size_same(s_h4, 2), conv_out_size_same(s_w4, 2)
            s_h16, s_w16 = conv_out_size_same(s_h8, 2), conv_out_size_same(s_w8, 2)

            h0 = linear(z, self.gf_dim*8*s_h16*s_w16, name='g_h0_lin')
            h0 = tf.reshape(h0, [-1, s_h16, s_w16, self.gf_dim * 8])
            h0 = tf.nn.relu(tf.layers.batch_normalization(h0))

            h1 = deconv2d(h0, [self.batch_size, s_h8, s_w8, self.gf_dim*4], name='g_h1')
            h1 = tf.nn.relu(tf.layers.batch_normalization(h1))

            h2 = deconv2d(h1, [self.batch_size, s_h4, s_w4, self.gf_dim*2], name='g_h2')
            h2 = tf.nn.relu(tf.layers.batch_normalization(h2))

            h3 = deconv2d(h2, [self.batch_size, s_h2, s_w2, self.gf_dim*1], name='g_h3')
            h3 = tf.nn.relu(tf.layers.batch_normalization(h3))

            h4 = deconv2d(h3, [self.batch_size, s_h, s_w, self.c_dim], name='g_h4')
            h4 = tf.nn.tanh(h4)

            tf.summary.histogram('gen/out', h4)
            tf.summary.image("gen", tf.reshape(h4, [-1, s_h, s_w, self.c_dim]), max_outputs=3)

            return h4

    def discriminator(self, image, reuse=False):
        '''
        h0 - (64, 32, 32, 64)
        h1 - (64, 16, 16, 128)
        h2 - (64, 8, 8, 256)
        h3 - (64, 4, 4, 512)
        h4 - (64, 1)
        '''
        with tf.variable_scope("discriminator", reuse=reuse):
            h0 = leaky_relu(conv2d(image, self.df_dim, name='d_h0_conv'))

            h1 = conv2d(h0, self.df_dim*2, name='d_h1_conv')
            h1 = leaky_relu(tf.layers.batch_normalization(h1))
            
            h2 = conv2d(h1, self.df_dim*4, name='d_h2_conv')
            h2 = leaky_relu(tf.layers.batch_normalization(h2))
            
            h3 = conv2d(h2, self.df_dim*8, name='d_h3_conv')
            h3 = leaky_relu(tf.layers.batch_normalization(h3))
            
            h4 = linear(tf.reshape(h3, [self.batch_size, -1]), 1, name='d_h4_lin')

            return tf.nn.sigmoid(h4), h4

    def build_model(self):
        self.z = tf.placeholder(tf.float32, [None, self.z_dim], name='z')
        self.inputs = tf.placeholder(tf.float32, [self.batch_size, \
            self.output_height, self.output_width, self.c_dim], name='real_images')

        self.G = self.generator(self.z)
        self.D, self.D_logits = self.discriminator(self.inputs, reuse=False)
        self.D_, self.D_logits_ = self.discriminator(self.G, reuse=True)
        # self.sampler = self.generator(self.z, reuse=True)

        self.g_loss = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.ones_like(self.D_)))

        self.d_loss_real = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits, labels=tf.ones_like(self.D)))
        self.d_loss_fake = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(logits=self.D_logits_, labels=tf.zeros_like(self.D_)))
        self.d_loss = self.d_loss_real + self.d_loss_fake

        tf.summary.scalar('g_loss', self.g_loss)
        tf.summary.scalar('d_loss', self.d_loss)

        self.d_vars = [var for var in tf.trainable_variables() if 'd_' in var.name]
        self.g_vars = [var for var in tf.trainable_variables() if 'g_' in var.name]

        self.d_optim = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5).minimize(self.d_loss, var_list=self.d_vars)
        self.g_optim = tf.train.AdamOptimizer(learning_rate=self.lr, beta1=0.5).minimize(self.g_loss, var_list=self.g_vars)
