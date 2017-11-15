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

net_rnn = rnn_embed(t_real_caption, is_train=False, reuse=False)

######### new stuff here
sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))

print("Loading weights from trained NN")
load_and_assign_npz(sess=sess, name=net_rnn_name, model=net_rnn)

######### 
'''
>>> captions_ids_train_decode_in[0]
[1, 152, 18, 32, 4, 8, 14, 711, 708]
>>> captions_ids_train_decode_out[0]
[152, 18, 32, 4, 8, 14, 711, 708, 2]
'''
captions_ids_train_decode_out = captions_ids_train ### [sentence, <end>]
captions_ids_train_decode_in = [] ### [<start>, sentence]
for caption_ids in captions_ids_train:
    caption_ids = caption_ids[:-1] ## getting rid of the </end>
    caption_ids = [vocab.start_id] + caption_ids
    captions_ids_train_decode_in.append(caption_ids)
captions_ids_train_decode_out = np.array(captions_ids_train_decode_out)
captions_ids_train_decode_in = np.array(captions_ids_train_decode_in)

decode_input = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name='embedding_decoding_caption_input')
decode_output = tf.placeholder(dtype=tf.int64, shape=[batch_size, None], name='embedding_decoding_caption_output')

decode_logits, decode_final_state = rnn_decoder(net_rnn.outputs, decode_input, is_train=True, reuse=False)
decode_loss_ = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(logits=decode_logits, labels=decode_output))


test_decode_input = tf.placeholder(dtype=tf.int64, shape=[1, None], name='embedding_decoding_caption_input_test')
test_initial_state = tf.placeholder(dtype=tf.float32, shape=[1, net_rnn.outputs.get_shape().as_list()[-1]], name="decoding_initial_state")
decode_test_logits, decode_test_final_state = rnn_decoder(test_initial_state, test_decode_input, is_train=False, reuse=True)


decode_vars = tl.layers.get_variables_with_name('rnndecodetxt', True, True)
decode_optim = tf.train.AdamOptimizer(0.01, beta1=0.5).minimize(decode_loss_, var_list=decode_vars )

tl.layers.initialize_global_variables(sess)

epoch = 8000
for i in range(epoch):
    idexs = get_random_int(min=0, max=n_captions_train-1, number=batch_size)
    b_decode_ins = captions_ids_train_decode_in[idexs]
    b_decode_outs = captions_ids_train_decode_out[idexs]
    b_decode_ins = tl.prepro.pad_sequences(b_decode_ins, padding='post')
    b_decode_outs = tl.prepro.pad_sequences(b_decode_outs, padding='post')
    errD, _ = sess.run([decode_loss_, decode_optim], 
                feed_dict={ t_real_caption : b_decode_outs,
                            decode_output: b_decode_outs,
                            decode_input : b_decode_ins})
    if i % 100 == 0:
        print("Epoch: [%2d/%2d]: decode_loss_: %.8f" % (i, epoch, errD))


#### see: https://github.com/zsdonghao/seq2seq-chatbot/blob/master/main_simple_seq2seq.py#L189
print "example is: "
print ' '.join(tl.nlp.word_ids_to_words(b_decode_outs[0], vocab.reverse_vocab))

for _ in range(5):  # 1 Query --> 5 Reply
    # 1. encode, get state
    state = sess.run(net_rnn.outputs,
                    {t_real_caption: b_decode_outs})
    state = state[[0]]
    # 3. decode, feed state iteratively
    sentence = [vocab.start_id]
    for _ in range(30):
        o, new_state = sess.run([decode_test_logits, decode_test_final_state],
                    {test_initial_state: state,
                    test_decode_input: [sentence]})
        w_id = tl.nlp.sample_top(o[-1][0], top_k=3)
        w = vocab.id_to_word(w_id)
        if w_id == vocab.end_id:
            break
        sentence = sentence + [w_id]
    print ' '.join(tl.nlp.word_ids_to_words(sentence, vocab.reverse_vocab))

