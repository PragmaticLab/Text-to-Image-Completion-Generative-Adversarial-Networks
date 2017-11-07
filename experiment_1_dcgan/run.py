import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import tensorflow as tf
import numpy as np 
from utils import *
from model import DCGAN


### Defining stuff ### 

BATCH_SIZE = 64
TOTAL_BATCHES = 1000000
INPUT_HEIGHT = INPUT_WIDTH = 64
OUTPUT_HEIGHT = OUTPUT_WIDTH = 64
Z_DIM = 100
GF_DIM = 128
DF_DIM = 64
C_DIM = 3
LR = 0.0005
LOGDIR = os.path.dirname(os.path.realpath(__file__)) + "/logs/"

### Loading ### 

X, _ = load_flowers(resize_pics=(INPUT_HEIGHT, INPUT_WIDTH))

dcgan = DCGAN(input_height=INPUT_HEIGHT, input_width=INPUT_WIDTH, batch_size=BATCH_SIZE,
            output_height=OUTPUT_HEIGHT, output_width=OUTPUT_WIDTH, z_dim=Z_DIM, gf_dim=GF_DIM, 
            df_dim=DF_DIM, c_dim=C_DIM, lr=LR)
dcgan.build_model()

### Training ### 

global_step = tf.Variable(0, name="global_step", trainable=False)
incr_global_step_op = tf.assign(global_step, global_step+1)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

checkpoint_file = os.path.join(LOGDIR, 'checkpoint')
saver = tf.train.Saver(max_to_keep=3)
checkpoint = tf.train.latest_checkpoint(LOGDIR)
summary = tf.summary.merge_all()
summary_writer = tf.summary.FileWriter(LOGDIR, sess.graph)


batches = batch_iter(X, batch_size=BATCH_SIZE, total_batches=TOTAL_BATCHES)
for batch in batches:
    if batch.shape[0] != BATCH_SIZE: continue 
    z = np.random.uniform(-1., 1., size=[batch.shape[0], Z_DIM])

    _, _, gl, dl, i, summary_str = sess.run(
        [dcgan.g_optim, dcgan.d_optim, dcgan.g_loss, dcgan.d_loss, incr_global_step_op, summary],
        feed_dict={dcgan.inputs: batch, dcgan.z: z})

    summary_writer.add_summary(summary_str, i)
    if i % 100 == 0 or i == 1:
        a = sess.run([dcgan.G], feed_dict={dcgan.inputs: batch, dcgan.z: z})[0]
        print a[0][0][0]
        print a[32][0][0]

        print('Step %i: Generator Loss: %f, Discriminator Loss: %f' % (i, gl, dl))
        saver.save(sess, checkpoint_file, global_step=global_step)

