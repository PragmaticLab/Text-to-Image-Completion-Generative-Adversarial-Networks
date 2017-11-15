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
                #+ tf.random_normal(shape=net_rnn.outputs.get_shape(), mean=0, stddev=0.02), # NOISE ON RNN
net_d, disc_fake_image_logits = discriminator_txt2img(
                net_fake_image.outputs, net_rnn.outputs, is_train=False, reuse=False)
_, disc_real_image_logits = discriminator_txt2img(
                    t_real_image, net_rnn.outputs, is_train=False, reuse=True)
d_loss1 = tl.cost.sigmoid_cross_entropy(disc_real_image_logits, tf.ones_like(disc_real_image_logits), name='d1')
net_g, _ = generator_txt2img(t_z,
                rnn_embed(t_real_caption, is_train=False, reuse=True).outputs,
                is_train=False, reuse=True, batch_size=batch_size)

######### new stuff here
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

print("Loading weights from trained NN")
load_and_assign_npz(sess=sess, name=net_rnn_name, model=net_rnn)
load_and_assign_npz(sess=sess, name=net_cnn_name, model=net_cnn)
load_and_assign_npz(sess=sess, name=net_g_name, model=net_g)
load_and_assign_npz(sess=sess, name=net_d_name, model=net_d)


# sample_size = batch_size
# sample_seed = np.random.normal(loc=0.0, scale=1.0, size=(sample_size, z_dim)).astype(np.float32)
# sample_sentence = ["the flower shown has yellow anther red pistil and bright red petals."] * int(sample_size/ni) + \
#                   ["this flower has petals that are yellow, white and purple and has dark lines"] * int(sample_size/ni) + \
#                   ["the petals on this flower are white with a yellow center"] * int(sample_size/ni) + \
#                   ["this flower has a lot of small round pink petals."] * int(sample_size/ni) + \
#                   ["this flower is orange in color, and has petals that are ruffled and rounded."] * int(sample_size/ni) + \
#                   ["the flower has yellow petals and the center of it is brown."] * int(sample_size/ni) + \
#                   ["this flower has petals that are blue and white."] * int(sample_size/ni) +\
#                   ["these white flowers have petals that start off white in color and end in a white towards the tips."] * int(sample_size/ni)
# for i, sentence in enumerate(sample_sentence):
#     print("seed: %s" % sentence)
#     sentence = preprocess_caption(sentence)
#     sample_sentence[i] = [vocab.word_to_id(word) for word in nltk.tokenize.word_tokenize(sentence)] + [vocab.end_id]

# sample_sentence = tl.prepro.pad_sequences(sample_sentence, padding='post')

# img_gen, rnn_out = sess.run([net_g.outputs, net_rnn.outputs], feed_dict={
#                                         t_real_caption : sample_sentence,
#                                         t_z : sample_seed})


lr_d = 0.0001
lr_rnn = 0.0005
beta1 = 0.5
epoch = 10000
l2_gamma = 0.1


idexs = get_random_int(min=0, max=n_captions_train-1, number=batch_size)
b_real_caption = captions_ids_train[idexs]
b_real_caption = tl.prepro.pad_sequences(b_real_caption, padding='post')
b_real_images = images_train[np.floor(np.asarray(idexs).astype('float')/n_captions_per_image).astype('int')]

## getting the real word embeddings that we learned
caption_embedding = tf.get_variable(name='caption_embedding3', shape=net_rnn.outputs.get_shape().as_list(), 
                    initializer=tf.random_normal_initializer(stddev=0.02))

alpha = 0.2 # margin alpha
rnn_caption_loss = tf.reduce_mean(tf.maximum(0., alpha - cosine_similarity(x, caption_embedding) + cosine_similarity(x, v_w))) + \
            tf.reduce_mean(tf.maximum(0., alpha - cosine_similarity(x, caption_embedding) + cosine_similarity(x_w, caption_embedding)))

_, disc_caption_logits = discriminator_txt2img(
                    t_real_image, caption_embedding, is_train=False, reuse=True)
d_caption_loss = tl.cost.sigmoid_cross_entropy(disc_caption_logits, tf.ones_like(disc_caption_logits), name='d_caption') + \
                    l2_gamma * tf.nn.l2_loss(caption_embedding)

rnn_caption_optim = tf.train.AdamOptimizer(lr_rnn, beta1=beta1).minimize(rnn_caption_loss, var_list=[caption_embedding])
d_caption_optim = tf.train.AdamOptimizer(lr_d, beta1=beta1).minimize(d_caption_loss, var_list=[caption_embedding])

eval_cosine_sim = tf.reduce_mean(cosine_similarity(caption_embedding, net_rnn.outputs))
eval_mse = tf.reduce_mean(tf.squared_difference(caption_embedding, net_rnn.outputs))

tl.layers.initialize_global_variables(sess)

actual_rnn_embedding = sess.run(net_rnn.outputs, feed_dict={t_real_caption : b_real_caption})
d_loss_real = sess.run(d_loss1, 
                feed_dict={
                    t_real_image : b_real_images,
                    t_real_caption: b_real_caption})
print "d_loss_real is %.8f" % (d_loss_real)

curr_best_errD, curr_best_errRNN = 5000, 5000
curr_best_embedding = None

for i in range(epoch):
    ## get wrong captions
    idexs = get_random_int(min=0, max=n_captions_train-1, number=batch_size)
    b_wrong_caption = captions_ids_train[idexs]
    b_wrong_caption = tl.prepro.pad_sequences(b_wrong_caption, padding='post')
    ## get wrong image
    idexs2 = get_random_int(min=0, max=n_images_train-1, number=batch_size)
    b_wrong_images = images_train[idexs2]
    errD, errRNN, _, _ = sess.run([d_caption_loss, rnn_caption_loss, d_caption_optim, rnn_caption_optim], 
                feed_dict={
                    t_real_image : b_real_images,
                    t_wrong_image: b_wrong_images,
                    t_wrong_caption : b_wrong_caption})
    if curr_best_errD > errD and curr_best_errRNN > errRNN:
        curr_best_embedding = sess.run(caption_embedding)
        curr_best_errD = errD
        curr_best_errRNN = errRNN
    if i % 100 == 0:
        cosine_sim_to_real, mse_to_real = sess.run([eval_cosine_sim, eval_mse], feed_dict={t_real_caption : b_real_caption})
        print("Epoch: [%2d/%2d]: d_caption_loss: %.8f, rnn_caption_loss: %.8f, mse_to_real: %.8f, cosine_sim_to_real: %.8f" % 
            (i, epoch, errD, errRNN, mse_to_real, cosine_sim_to_real))

print curr_best_errD, curr_best_errRNN

# ####### generate image instead

# lr = 0.01
# beta1 = 0.5
# epoch = 30

# idexs = get_random_int(min=0, max=n_captions_train-1, number=batch_size)
# b_real_caption = captions_ids_train[idexs]
# b_real_caption = tl.prepro.pad_sequences(b_real_caption, padding='post')
# b_real_images = images_train[np.floor(np.asarray(idexs).astype('float')/n_captions_per_image).astype('int')]

# ## getting the real word embeddings that we learned
# new_img = tf.get_variable(name='new_img_LOL', shape=t_real_image.get_shape().as_list(), 
#                     initializer=tf.random_normal_initializer(stddev=0.02))
# _, disc_caption_logits = discriminator_txt2img(
#                     new_img, net_rnn.outputs, is_train=False, reuse=True) # make sure that is_train is False

# d_caption_loss = tl.cost.sigmoid_cross_entropy(disc_caption_logits, tf.ones_like(disc_caption_logits), name='d_caption')
# d_optim = tf.train.AdamOptimizer(lr, beta1=beta1).minimize(d_caption_loss, var_list=[new_img])

# tl.layers.initialize_global_variables(sess)


# for i in range(epoch):
#     errD, _ = sess.run([d_caption_loss, d_optim], feed_dict={
#                     t_real_caption : b_real_caption})
#     if i % 100 == 0:
#         generated_new_img = sess.run(new_img)
#         # mse = ((generated_caption_embedding - actual_rnn_embedding) ** 2).mean()
#         print("Epoch: [%2d/%2d]: d_caption_loss: %.8f" % 
#             (i, epoch, errD))

# save_images(b_real_images, [ni, ni], 'samples/jason_original.png')
# save_images(generated_new_img, [ni, ni], 'samples/jason_generated.png')
