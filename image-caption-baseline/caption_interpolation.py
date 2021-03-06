#!/usr/bin/env python
"""
Code description
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import os.path as osp
import tensorflow as tf
from lm import lm_loss, build_lm, MAX_GRADIENT_NORM, RNN_SIZE, MAX_SEQ_LEN, build_sampler
from cnn import encode_image
from gen_vocab import load_vocab, EOS, print_text, BOS
from coco_inputs import inputs
from time import gmtime, strftime
import numpy as np

LEARNING_RATE = 1e-3

def _lm_sampler(images, num_symbols, batch_size, paths_):
  cnn = encode_image(images)
  samples = build_sampler(num_symbols, cnn, batch_size)

  def sample(sess):
    samples_, _paths = sess.run([samples, paths_])
    return samples_, _paths

  return sample

def simple_softmax(x):
  ps = np.exp(x)
  ps /= (np.sum(ps))
  ps -= 0.0000001
  return ps

def _cnn_sampler(images1, images2, num_symbols, batch_size, paths_):
  cnn1 = encode_image(images1)
  cnn2 = encode_image(images2, reuse=True)
  cnn = 0.5 * cnn1 + 0.5 * cnn2

  prev_state = tf.placeholder(tf.float32, [None, RNN_SIZE])
  prev_symbol = tf.placeholder(tf.int32, [None, 1])

  one_step = build_lm(prev_symbol, num_symbols,
      prev_state, seq_len=1)

  initial_text = np.ones([batch_size,1], dtype=np.int32) * BOS

  def sample(sess):
    _cnn, _paths = sess.run([cnn, paths_])
    prev = _cnn
    text = initial_text
    texts = []
    mask = np.zeros(batch_size, dtype=bool)

    for step in range(MAX_SEQ_LEN):
      logits, prev = sess.run(one_step,
          feed_dict={
            prev_state:prev,
            prev_symbol:text})
      # print(logits.shape)
      text = np.argmax(logits, 1) # maybe use np.random.multinomial(1, [0.01, 0.99])
      # print(np.sum(simple_softmax(logits[0])))
      # text = [np.argmax(np.random.multinomial(1, simple_softmax(logits[i]))) for i in range(batch_size)]
      text = np.expand_dims(text, 1)
      texts.append(text)

      mask = np.logical_or(mask, text == EOS)
      if mask.all():
        break

    texts = np.concatenate(texts, axis=1)
    return texts, _paths

  return sample

def image2text(images1, images2, captions, num_symbols):
  cnn1 = encode_image(images1)
  cnn2 = encode_image(images2, reuse=True)
  cnn = (cnn1 + cnn2) / 2
  loss = lm_loss(captions, num_symbols, cnn)
  return loss

def main():
  data_dir1 = sys.argv[1]
  data_dir2 = sys.argv[2]
  vocab_path = sys.argv[3]
  ckpt_path = sys.argv[4]

  do_train = True
  try:
    eval_save_path = sys.argv[5]
    do_train = eval_save_path == 'train'
  except:
    pass

  batch_size = 1

  _, i2w = load_vocab(vocab_path)
  num_symbols = len(i2w)
  print('num_symbols:', num_symbols)
  with tf.Graph().as_default():
    sess = tf.Session()
    with tf.device('/cpu:0'):
      images1, captions, coco_ids = inputs(data_dir1,
          do_train,
          batch_size,
          None if do_train else 1)
      images2, captions, coco_ids = inputs(data_dir2,
          do_train,
          batch_size,
          None if do_train else 1)

    with tf.variable_scope("im2txt"):
      loss = image2text(images1, images2, captions, num_symbols)

    with tf.variable_scope("im2txt", reuse=True):
      sampler = _cnn_sampler(images1, images2, num_symbols, batch_size, coco_ids)

    params = tf.trainable_variables()
    opt = tf.train.AdamOptimizer(LEARNING_RATE)

    gradients = tf.gradients(loss, params)
    clipped_gradients, norm = tf.clip_by_global_norm(gradients,
                                                     MAX_GRADIENT_NORM)

    global_step = tf.Variable(0, trainable=False)
    train_op = opt.apply_gradients(
        zip(clipped_gradients, params), global_step=global_step)

    # Create a saver.
    saver = tf.train.Saver(tf.all_variables())

    init_op = tf.group(tf.initialize_all_variables(),
                  tf.initialize_local_variables())

    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    tf.get_default_graph().finalize()

    if osp.exists(ckpt_path):
      saver.restore(sess, ckpt_path)

    start = global_step.eval(session=sess)

    def _eval(output=sys.stdout):
      samples_, paths_ = sampler(sess)
      print_text(samples_, i2w, paths_, file=output)

    def eval():
      save_path = eval_save_path + '-%d' % start
      if osp.exists(save_path):
        return
      try:
        with open(save_path, 'w') as writer:
          cnt = 0
          while not coord.should_stop():
            _eval(writer)
            if cnt % 100 == 0:
              print(cnt, strftime("%Y-%m-%d %H:%M:%S", gmtime()))
            cnt += 1
      except tf.errors.OutOfRangeError:
        print('finish eval')

    eval()

    coord.request_stop()
    coord.join(threads)
    sess.close()

if __name__ == "__main__":
  main()

# vim: tabstop=4 expandtab shiftwidth=2 softtabstop=2
