import tensorflow as tf
import tensorlayer as tl
from tensorlayer.layers import *
from tensorlayer.prepro import *
from tensorlayer.cost import *
import numpy as np
import scipy
from scipy.io import loadmat
import time, os, re, nltk

from utils import *
from model import *
import model

print("Loading data from pickle ...")
import pickle
with open("_vocab.pickle", 'rb') as f:
    vocab = pickle.load(f)
with open("_image_train.pickle", 'rb') as f:
    _, images_train = pickle.load(f)
with open("_image_test.pickle", 'rb') as f:
    _, images_test = pickle.load(f)
with open("_n.pickle", 'rb') as f:
    n_captions_train, n_captions_test, n_captions_per_image, n_images_train, n_images_test = pickle.load(f)
with open("_caption.pickle", 'rb') as f:
    captions_ids_train, captions_ids_test = pickle.load(f)
images_train = np.array(images_train)
images_test = np.array(images_test)


save_dir = "checkpoint"
net_rnn_name = os.path.join(save_dir, 'net_rnn.npz')
net_cnn_name = os.path.join(save_dir, 'net_cnn.npz')
net_g_name = os.path.join(save_dir, 'net_g.npz')
net_d_name = os.path.join(save_dir, 'net_d.npz')
ni = int(np.ceil(np.sqrt(batch_size)))

t_real_image = tf.placeholder('float32', [batch_size, image_size, image_size, 3], name = 'real_image')
t_real_caption = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name='real_caption_input')
t_z = tf.placeholder(tf.float32, [batch_size, z_dim], name='z_noise')

net_cnn = cnn_encoder(t_real_image, is_train=False, reuse=False)
x = net_cnn.outputs
net_rnn = rnn_embed(t_real_caption, is_train=False, reuse=False)
v = net_rnn.outputs

######### new stuff here
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

lr = 0.0002
beta1 = 0.5
n_epoch = 100
n_batch_epoch = int(n_images_train / batch_size)


w_init = tf.random_normal_initializer(stddev=0.02)
network = tl.layers.DenseLayer(net_cnn, n_units=800, W_init = w_init,
                                act = tf.nn.relu, name='cycle/relu1')
network = tl.layers.DropoutLayer(network, keep=0.5, is_fix=True, name='cycle/drop1')
network = tl.layers.DenseLayer(network, n_units=400, W_init = w_init,
                                act = tf.nn.relu, name='cycle/relu2')
network = tl.layers.DropoutLayer(network, keep=0.5, is_fix=True, name='cycle/drop2')
network = tl.layers.DenseLayer(network, n_units=128, W_init = w_init,
                                act = tf.nn.relu, name='cycle/relu3')
cost = tl.cost.mean_squared_error(network.outputs, net_rnn.outputs, is_mean=False)


cycle_vars = tl.layers.get_variables_with_name('cycle', True, True)
optim = tf.train.AdamOptimizer(lr, beta1=beta1).minimize(cost, var_list=cycle_vars)


tl.layers.initialize_global_variables(sess)
load_and_assign_npz(sess=sess, name=net_rnn_name, model=net_rnn)
load_and_assign_npz(sess=sess, name=net_cnn_name, model=net_cnn)


for epoch in range(0, n_epoch+1):
    for step in range(n_batch_epoch):
        idexs = get_random_int(min=0, max=n_captions_train-1, number=batch_size)
        b_real_caption = captions_ids_train[idexs]
        b_real_caption = tl.prepro.pad_sequences(b_real_caption, padding='post')
        b_real_images = images_train[np.floor(np.asarray(idexs).astype('float')/n_captions_per_image).astype('int')]
        b_real_images = threading_data(b_real_images, prepro_img, mode='train')   # [0, 255] --> [-1, 1] + augmentation
        b_cost, _ = sess.run([cost, optim], feed_dict={
                                                t_real_image : b_real_images,
                                                t_real_caption : b_real_caption})
        if step % 500 == 0:
            print "epoch:%d, step:%d, cost: %.4f" % (epoch, step, b_cost)



a = sess.run([net_rnn.outputs], feed_dict={
                                                t_real_image : b_real_images,
                                                t_real_caption : b_real_caption})



