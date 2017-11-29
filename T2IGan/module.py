from __future__ import division
import tensorflow as tf
from ops import *
from utils import *
import tensorlayer as tl 
from tensorlayer.layers import *

t_dim = 128         # text feature dimension
rnn_hidden_size = t_dim
vocab_size = 8000
word_embedding_size = 256
keep_prob = 1.0

def rnn_embed(input_seqs, is_train=True, reuse=False, return_embed=False):
    """ txt --> t_dim """
    w_init = tf.random_normal_initializer(stddev=0.02)
    if tf.__version__ <= '0.12.1':
        LSTMCell = tf.nn.rnn_cell.LSTMCell
    else:
        LSTMCell = tf.contrib.rnn.BasicLSTMCell
    with tf.variable_scope("rnnftxt", reuse=reuse):
        tl.layers.set_name_reuse(reuse)
        network = EmbeddingInputlayer(
                     inputs = input_seqs,
                     vocabulary_size = vocab_size,
                     embedding_size = word_embedding_size,
                     E_init = w_init,
                     name = 'rnn/wordembed')
        network = DynamicRNNLayer(network,
                     cell_fn = LSTMCell,
                     cell_init_args = {'state_is_tuple' : True, 'reuse': reuse},  # for TF1.1, TF1.2 dont need to set reuse
                     n_hidden = rnn_hidden_size,
                     dropout = (keep_prob if is_train else None),
                     initializer = w_init,
                     sequence_length = tl.layers.retrieve_seq_length_op2(input_seqs),
                     return_last = True,
                     name = 'rnn/dynamic')
        return network


def discriminator(image, text, options, reuse=False, name="discriminator"):

    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        h0 = lrelu(conv2d(image, options.df_dim, name='d_h0_conv'))
        # h0 is (128 x 128 x self.df_dim)
        h1 = lrelu(instance_norm(conv2d(h0, options.df_dim*2, name='d_h1_conv'), 'd_bn1'))
        # h1 is (64 x 64 x self.df_dim*2)
        h2 = lrelu(instance_norm(conv2d(h1, options.df_dim*4, name='d_h2_conv'), 'd_bn2'))
        # h2 is (32x 32 x self.df_dim*4)
        h3 = lrelu(instance_norm(conv2d(h2, options.df_dim*8, s=1, name='d_h3_conv'), 'd_bn3'))
        # h3 is (32 x 32 x self.df_dim*8)

        w_init = tf.random_normal_initializer(stddev=0.02)
        gamma_init=tf.random_normal_initializer(1., 0.02)
        net_h3 = InputLayer(h3, name='d_input_img')
        net_txt = InputLayer(text, name='d_input_txt')
        net_txt = DenseLayer(net_txt, n_units=t_dim,
               act=lambda x: tl.act.lrelu(x, 0.2),
               W_init=w_init, name='d_reduce_txt/dense')
        net_txt = ExpandDimsLayer(net_txt, 1, name='d_txt/expanddim1')        
        net_txt = ExpandDimsLayer(net_txt, 1, name='d_txt/expanddim2')
        net_txt = TileLayer(net_txt, [1, 32, 32, 1], name='d_txt/tile')
        net_h3_concat = ConcatLayer([net_h3, net_txt], concat_dim=3, name='d_h3_concat')
        h4 = conv2d(net_h3_concat.outputs, 1, s=1, name='d_h3_pred')

        return h4
        # return h3


def generator_resnet(image, text, options, reuse=False, name="generator"):

    with tf.variable_scope(name):
        # image is 256 x 256 x input_c_dim
        if reuse:
            tf.get_variable_scope().reuse_variables()
        else:
            assert tf.get_variable_scope().reuse is False

        def residule_block(x, dim, ks=3, s=1, name='res'):
            p = int((ks - 1) / 2)
            y = tf.pad(x, [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_c1'), name+'_bn1')
            y = tf.pad(tf.nn.relu(y), [[0, 0], [p, p], [p, p], [0, 0]], "REFLECT")
            y = instance_norm(conv2d(y, dim, ks, s, padding='VALID', name=name+'_c2'), name+'_bn2')
            return y + x

        # Justin Johnson's model from https://github.com/jcjohnson/fast-neural-style/
        # The network with 9 blocks consists of: c7s1-32, d64, d128, R128, R128, R128,
        # R128, R128, R128, R128, R128, R128, u64, u32, c7s1-3
        c0 = tf.pad(image, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        c1 = tf.nn.relu(instance_norm(conv2d(c0, options.gf_dim, 7, 1, padding='VALID', name='g_e1_c'), 'g_e1_bn'))
        c2 = tf.nn.relu(instance_norm(conv2d(c1, options.gf_dim*2, 3, 2, name='g_e2_c'), 'g_e2_bn'))
        c3 = tf.nn.relu(instance_norm(conv2d(c2, options.gf_dim*4, 3, 2, name='g_e3_c'), 'g_e3_bn'))
        c4 = tf.nn.relu(instance_norm(conv2d(c3, options.gf_dim*4, 3, 2, name='g_e4_c'), 'g_e4_bn')) # 32x32x256
        c5 = tf.nn.relu(instance_norm(conv2d(c4, options.gf_dim*4, 3, 2, name='g_e5_c'), 'g_e5_bn')) # 16x16x256

        w_init = tf.random_normal_initializer(stddev=0.02)
        gamma_init=tf.random_normal_initializer(1., 0.02)
        net_c5 = InputLayer(c5, name='g_input_img')
        net_txt = InputLayer(text, name='g_input_txt')
        net_txt = DenseLayer(net_txt, n_units=t_dim,
               act=lambda x: tl.act.lrelu(x, 0.2),
               W_init=w_init, name='g_reduce_txt/dense')
        net_txt = ExpandDimsLayer(net_txt, 1, name='g_txt/expanddim1')        
        net_txt = ExpandDimsLayer(net_txt, 1, name='g_txt/expanddim2')
        net_txt = TileLayer(net_txt, [1, 16, 16, 1], name='g_txt/tile')
        net_c5_concat = ConcatLayer([net_c5, net_txt], concat_dim=3, name='g_h3_concat')
        c6 = net_c5_concat.outputs

        # define G network with 9 resnet blocks
        r1 = residule_block(c6, options.gf_dim*4+t_dim, name='g_r1')
        r2 = residule_block(r1, options.gf_dim*4+t_dim, name='g_r2')
        r3 = residule_block(r2, options.gf_dim*4+t_dim, name='g_r3')
        r4 = residule_block(r3, options.gf_dim*4+t_dim, name='g_r4')
        r5 = residule_block(r4, options.gf_dim*4+t_dim, name='g_r5')
        r6 = residule_block(r5, options.gf_dim*4+t_dim, name='g_r6')
        r7 = residule_block(r6, options.gf_dim*4+t_dim, name='g_r7')
        r8 = residule_block(r7, options.gf_dim*4+t_dim, name='g_r8')
        r9 = residule_block(r8, options.gf_dim*4+t_dim, name='g_r9')

        d1 = deconv2d(r9, options.gf_dim*4, 3, 2, name='g_d1_dc')
        d1 = tf.nn.relu(instance_norm(d1, 'g_d1_bn'))
        d2 = deconv2d(d1, options.gf_dim*2, 3, 2, name='g_d2_dc')
        d2 = tf.nn.relu(instance_norm(d2, 'g_d2_bn'))
        d3 = deconv2d(d2, options.gf_dim*2, 3, 2, name='g_d3_dc')
        d3 = tf.nn.relu(instance_norm(d3, 'g_d3_bn'))
        d4 = deconv2d(d3, options.gf_dim*1, 3, 2, name='g_d4_dc')
        d4 = tf.nn.relu(instance_norm(d4, 'g_d4_bn'))
        d4 = tf.pad(d4, [[0, 0], [3, 3], [3, 3], [0, 0]], "REFLECT")
        pred = tf.nn.tanh(conv2d(d4, options.output_c_dim, 7, 1, padding='VALID', name='g_pred_c'))

        return pred


def abs_criterion(in_, target):
    return tf.reduce_mean(tf.abs(in_ - target))


def mae_criterion(in_, target):
    return tf.reduce_mean((in_-target)**2)


def sce_criterion(logits, labels):
    return tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
