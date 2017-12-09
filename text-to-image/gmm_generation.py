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

from sklearn.mixture import GaussianMixture


print("Loading data from pickle ...")
import pickle
with open("_vocab.pickle", 'rb') as f:
    vocab = pickle.load(f)
# with open("_image_train.pickle", 'rb') as f:
#     _, images_train = pickle.load(f)
# with open("_image_test.pickle", 'rb') as f:
#     _, images_test = pickle.load(f)
with open("_n.pickle", 'rb') as f:
    n_captions_train, n_captions_test, n_captions_per_image, n_images_train, n_images_test = pickle.load(f)
with open("_caption.pickle", 'rb') as f:
    captions_ids_train, captions_ids_test = pickle.load(f)
# images_train = np.array(images_train)
# images_test = np.array(images_test)

save_dir = "checkpoint"
net_rnn_name = os.path.join(save_dir, 'net_rnn.npz')
net_cnn_name = os.path.join(save_dir, 'net_cnn.npz')
net_g_name = os.path.join(save_dir, 'net_g.npz')
net_d_name = os.path.join(save_dir, 'net_d.npz')
ni = int(np.ceil(np.sqrt(batch_size)))

t_real_image = tf.placeholder('float32', [batch_size, image_size, image_size, 3], name = 'real_image')
t_wrong_image = tf.placeholder('float32', [batch_size ,image_size, image_size, 3], name = 'wrong_image')
t_real_caption = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name='real_caption_input')
t_wrong_caption = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name='wrong_caption_input')
t_z = tf.placeholder(tf.float32, [batch_size, z_dim], name='z_noise')

net_cnn = cnn_encoder(t_real_image, is_train=False, reuse=False)
x = net_cnn.outputs
v = rnn_embed(t_real_caption, is_train=False, reuse=False).outputs
x_w = cnn_encoder(t_wrong_image, is_train=False, reuse=True).outputs
v_w = rnn_embed(t_wrong_caption, is_train=False, reuse=True).outputs

generator_txt2img = model.generator_txt2img_resnet
discriminator_txt2img = model.discriminator_txt2img_resnet

net_rnn = rnn_embed(t_real_caption, is_train=False, reuse=True)
net_fake_image, _ = generator_txt2img(t_z,
                net_rnn.outputs,
                is_train=False, reuse=False, batch_size=batch_size)
net_g, _ = generator_txt2img(t_z,
                rnn_embed(t_real_caption, is_train=False, reuse=True).outputs,
                is_train=False, reuse=True, batch_size=batch_size)

embedding = tf.placeholder(dtype='float32', shape=[batch_size, 128])
generator, _ = generator_txt2img(t_z,
                embedding,
                is_train=False, reuse=True, batch_size=batch_size)


######### new stuff here
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

# t_caption1 = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name='caption1')
# t_caption2 = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name='caption2')

# caption1 = rnn_embed(t_caption1, is_train=False, reuse=True).outputs
# caption2 = rnn_embed(t_caption2, is_train=False, reuse=True).outputs
# merged_caption = (caption1 + caption2) / 2
# # merged_caption = caption1 + caption2 * 0

# merged_image, _ = generator_txt2img(t_z, merged_caption,
#                 is_train=False, reuse=True, batch_size=batch_size)

print("Loading weights from trained NN")
load_and_assign_npz(sess=sess, name=net_rnn_name, model=net_rnn)
load_and_assign_npz(sess=sess, name=net_cnn_name, model=net_cnn)
load_and_assign_npz(sess=sess, name=net_g_name, model=net_g)

sample_size = batch_size
sample_seed = np.random.normal(loc=0.0, scale=1.0, size=(sample_size, z_dim)).astype(np.float32)

captions = tl.prepro.pad_sequences(captions_ids_train, padding='post')

list_of_embeddings = []
for i in range(captions.shape[0] / 64):
    caption_batch = captions[i*64:(i+1)*64]
    caption_embeddings = sess.run(v, feed_dict={t_real_caption: caption_batch})
    list_of_embeddings.append(caption_embeddings)

all_embeddings = np.concatenate(list_of_embeddings, axis=0) 

total_components = 120
gmm = GaussianMixture(n_components=total_components, covariance_type='diag', verbose=5, max_iter=500)
gmm.fit(all_embeddings)

x, y = gmm.sample(64000)
for i in range(total_components): # 64 images for each cluster in GMM 
    x_i = x[y==i]
    x_i = x_i[:64]
    img_gen = sess.run(generator.outputs, feed_dict={
                                            embedding : x_i,
                                            t_z : sample_seed})
    save_images(img_gen, [ni, ni], 'samples/gmm_generated/%d.png' % i)
