from flask import Flask, jsonify, request, send_file

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

######### new stuff here
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

t_caption1 = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name='caption1')
caption1 = rnn_embed(t_caption1, is_train=False, reuse=True).outputs
image, _ = generator_txt2img(t_z, caption1,
                is_train=False, reuse=True, batch_size=batch_size)

load_and_assign_npz(sess=sess, name=net_rnn_name, model=net_rnn)
load_and_assign_npz(sess=sess, name=net_cnn_name, model=net_cnn)
load_and_assign_npz(sess=sess, name=net_g_name, model=net_g)

sample_size = batch_size
sample_seed = np.random.normal(loc=0.0, scale=1.0, size=(sample_size, z_dim)).astype(np.float32)


def get_pad_seq(samples):
    for i, sentence in enumerate(samples):
        sentence = preprocess_caption(sentence)
        samples[i] = [vocab.word_to_id(word) for word in nltk.tokenize.word_tokenize(sentence)] + [vocab.end_id]
    samples = tl.prepro.pad_sequences(samples, padding='post')
    return samples

def process_vocab(sample):
    sample_batch = [sample] * sample_size
    return get_pad_seq(sample_batch)


app = Flask(__name__)

@app.route('/caption_to_img')
def caption_to_img():
    caption = request.args.get('caption')
    caption = process_vocab(caption)
    [generated_imgs] = sess.run([image.outputs], feed_dict={
                                            t_caption1 : caption,
                                            t_z : sample_seed})
    save_images(generated_imgs, [ni, ni], 'samples/demo_server.png')
    return send_file('samples/demo_server.png', mimetype='image/gif')

app.run() 

'''http://127.0.0.1:5000/caption_to_img?caption=this%20flower%20is%20red'''


