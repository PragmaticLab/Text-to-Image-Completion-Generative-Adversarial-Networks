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
import scipy.misc

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
discriminator, disc_fake_image_logits = discriminator_txt2img(
                generator.outputs, embedding, is_train=True, reuse=False)

######### new stuff here
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

print("Loading weights from trained NN")
load_and_assign_npz(sess=sess, name=net_rnn_name, model=net_rnn)
load_and_assign_npz(sess=sess, name=net_cnn_name, model=net_cnn)
load_and_assign_npz(sess=sess, name=net_g_name, model=net_g)
load_and_assign_npz(sess=sess, name=net_d_name, model=discriminator)


sample_size = batch_size

with open("gmm_embedding.pickle", 'rb') as f:
    gmm = pickle.load(f)

samples,_ = gmm.sample(100000)
scores = gmm.score_samples(samples)
worst_k = scores.argsort()[:np.sum(scores < 330)]
best_k = scores.argsort()[-np.sum(scores > 500):]
worst_k = np.random.choice(worst_k, size=64)
best_k = np.random.choice(best_k, size=64)

print np.mean(scores[worst_k])
print np.mean(scores[best_k])

# worst_k = best_k

sample_seed = np.random.normal(loc=0.0, scale=1.0, size=(sample_size, z_dim)).astype(np.float32)
img_gen, img_score = sess.run([generator.outputs, discriminator.outputs], feed_dict={
                                        embedding : samples[worst_k],
                                        t_z : sample_seed})

for i in range(img_gen.shape[0]):
    img = scipy.misc.imresize(img_gen[i], size=[500, 500])
    scipy.misc.imsave('samples/bad_gmm_samples/image_%d.jpg' % (10000 + i), img)
