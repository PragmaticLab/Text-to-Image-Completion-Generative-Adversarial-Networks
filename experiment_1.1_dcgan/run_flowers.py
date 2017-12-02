import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import *

X, _ = load_flowers(resize_pics=(32, 32))

# Training Params
num_steps = 500000
batch_size = 32
lr_generator = 0.0003
lr_discriminator = 0.0001

# Network Params
image_dim = 784 # 28*28 pixels * 3 channel
noise_dim = 100 # Noise data points# Build Networks


# Build Networks
# Network Inputs
noise_input = tf.placeholder(tf.float32, shape=[None, noise_dim])
real_image_input = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
# A boolean to indicate batch normalization if it is training or inference time
is_training = tf.placeholder(tf.bool)

def leakyrelu(x, alpha=0.02):
    return tf.maximum(x, alpha*x)

w_init = tf.random_normal_initializer(stddev=0.02)
gamma_init = tf.random_normal_initializer(1., 0.02)

def generator(x, reuse=False):
    with tf.variable_scope('Generator', reuse=reuse):
        x = tf.layers.dense(x, units=4*4*64, kernel_initializer=w_init)
        x = tf.layers.batch_normalization(x, training=is_training, gamma_initializer=gamma_init)
        x = tf.nn.relu(x)
        x = tf.reshape(x, shape=[-1, 4, 4, 64])
        x = tf.layers.conv2d_transpose(x, 32, 2, strides=2, padding='same', kernel_initializer=w_init)
        x = tf.layers.batch_normalization(x, training=is_training, gamma_initializer=gamma_init)
        x = tf.nn.relu(x)
        x = tf.layers.conv2d_transpose(x, 16, 2, strides=2, padding='same', kernel_initializer=w_init)
        x = tf.layers.batch_normalization(x, training=is_training, gamma_initializer=gamma_init)
        x = tf.nn.relu(x)
        x = tf.layers.conv2d_transpose(x, 3, 2, strides=2, padding='same', kernel_initializer=w_init)
        x = tf.nn.tanh(x)
        return x


def discriminator(x, reuse=False):
    with tf.variable_scope('Discriminator', reuse=reuse):
        x = tf.layers.conv2d(x, 16, 2, strides=2, padding='same', kernel_initializer=w_init)
        x = tf.layers.batch_normalization(x, training=is_training, gamma_initializer=gamma_init)
        x = leakyrelu(x)
        x = tf.layers.conv2d(x, 32, 2, strides=2, padding='same', kernel_initializer=w_init)
        x = tf.layers.batch_normalization(x, training=is_training, gamma_initializer=gamma_init)
        x = leakyrelu(x)
        x = tf.layers.conv2d(x, 64, 2, strides=2, padding='same', kernel_initializer=w_init)
        x = tf.layers.batch_normalization(x, training=is_training, gamma_initializer=gamma_init)
        x = leakyrelu(x)
        x = tf.reshape(x, shape=[-1, 4*4*64])
        x = tf.layers.dense(x, 1024, kernel_initializer=w_init)
        x = tf.layers.batch_normalization(x, training=is_training, gamma_initializer=gamma_init)
        x = leakyrelu(x)
        x = tf.layers.dense(x, 2, kernel_initializer=w_init)
    return x

# Build Generator Network
gen_sample = generator(noise_input)

# Build 2 Discriminator Networks (one from noise input, one from generated samples)
disc_real = discriminator(real_image_input)
disc_fake = discriminator(gen_sample, reuse=True)

# Build the stacked generator/discriminator
stacked_gan = discriminator(gen_sample, reuse=True)

# Build Loss (Labels for real images: 1, for fake images: 0)
# Discriminator Loss for real and fake samples
disc_loss_real = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=disc_real, labels=tf.ones([batch_size], dtype=tf.int32)))
disc_loss_fake = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=disc_fake, labels=tf.zeros([batch_size], dtype=tf.int32)))
# Sum both loss
disc_loss = disc_loss_real + disc_loss_fake
# Generator Loss (The generator tries to fool the discriminator, thus labels are 1)
gen_loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(
    logits=stacked_gan, labels=tf.ones([batch_size], dtype=tf.int32)))

# Build Optimizers
optimizer_gen = tf.train.AdamOptimizer(learning_rate=lr_generator)
optimizer_disc = tf.train.AdamOptimizer(learning_rate=lr_discriminator)

# Training Variables for each optimizer
# By default in TensorFlow, all variables are updated by each optimizer, so we
# need to precise for each one of them the specific variables to update.
# Generator Network Variables
gen_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Generator')
# Discriminator Network Variables
disc_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='Discriminator')

# Create training operations
# TensorFlow UPDATE_OPS collection holds all batch norm operation to update the moving mean/stddev
gen_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Generator')
# `control_dependencies` ensure that the `gen_update_ops` will be run before the `minimize` op (backprop)
with tf.control_dependencies(gen_update_ops):
    train_gen = optimizer_gen.minimize(gen_loss, var_list=gen_vars)
disc_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, scope='Discriminator')
with tf.control_dependencies(disc_update_ops):
    train_disc = optimizer_disc.minimize(disc_loss, var_list=disc_vars)
    
# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()


# Start Training
# Start a new TF session
sess = tf.Session()

# Run the initializer
sess.run(init)


batches = batch_iter(X, batch_size=batch_size, total_batches=num_steps)
dl = 999.0
for i, batch in enumerate(batches):
    if batch.shape[0] != batch_size: continue 

    z = np.random.normal(loc=0.0, scale=1.0, size=[batch_size, noise_dim]).astype(np.float32)

    if dl > 0.2:
        _, dl, _, gl = sess.run([train_disc, disc_loss, train_gen, gen_loss], 
            feed_dict={real_image_input: batch, noise_input: z, is_training:True})
    dl, _, gl = sess.run([disc_loss, train_gen, gen_loss], 
        feed_dict={real_image_input: batch, noise_input: z, is_training:True})

    if i % 100 == 0:
        g = sess.run(gen_sample, feed_dict={noise_input: z, is_training:False})
        plt.imshow(g[0])
        plt.savefig('experiment_1.1_dcgan/samples/sample_%d.png' % i)
        print('Step %i: Generator Loss: %f, Discriminator Loss: %f' % (i, gl, dl))
