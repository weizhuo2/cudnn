# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""A binary to train CIFAR-10 using a single GPU.

Accuracy:
cifar10_train.py achieves ~86% accuracy after 100K steps (256 epochs of
data) as judged by cifar10_eval.py.

Speed: With batch_size 128.

System        | Step Time (sec/batch)  |     Accuracy
------------------------------------------------------------------
1 Tesla K20m  | 0.35-0.60              | ~86% at 60K steps  (5 hours)
1 Tesla K40m  | 0.25-0.35              | ~86% at 100K steps (4 hours)

Usage:
Please see the tutorial and website for how to download the CIFAR-10
data set, compile the program and train the model.

http://tensorflow.org/tutorials/deep_cnn/
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from datetime import datetime
import time

import tensorflow as tf

import cifar10

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('train_dir', '/tmp/cifar10_train',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_integer('max_steps', 4000,
                            """Number of batches to run.""")
tf.app.flags.DEFINE_boolean('log_device_placement', False,
                            """Whether to log device placement.""")
tf.app.flags.DEFINE_integer('log_frequency', 100,
                            """How often to log results to the console.""")


def train():
  """Train CIFAR-10 for a number of steps."""
  with tf.Graph().as_default():
    # with tf.variable_scope("cifar10", reuse=tf.AUTO_REUSE) as scope:
    global_step = tf.train.get_or_create_global_step()

    # Get images and labels for CIFAR-10.
    train_flag = tf.placeholder(tf.bool, shape = ())

    trX, trY = cifar10.distorted_inputs()
    teX, teY = cifar10.inputs(eval_data = True)

    # Build a Graph that computes the logits predictions from the
    # inference model.
    logits = cifar10.inference(trX)
    # Calculate accuracy
    tr_acc = cifar10.accuracy(logits, trY)[1]
    print(tr_acc, "tr_acc\n")
    # tr_acc_sum = tf.summary.scalar('train/accuracy', tr_acc)
    # Calculate loss.
    loss = cifar10.loss(logits, trY)
    # Build a Graph that trains the model with one batch of examples and
    # updates the model parameters.
    train_op = cifar10.train(loss, global_step)

    tf.get_variable_scope().reuse_variables()
    eval_logits = cifar10.inference(teX)
    te_acc = cifar10.accuracy(eval_logits, teY)[1]
    print(te_acc, "te_acc\n")
    # te_acc_sum = tf.summary.scalar('test/accuracy', te_acc)

    accuracy = tf.cond(train_flag, lambda: tr_acc, lambda: te_acc)
    tf.summary.scalar("accuracy", accuracy)

    merged = tf.summary.merge_all()
    train_writer = tf.summary.FileWriter('tmp/cifar10/train')
    test_writer = tf.summary.FileWriter('tmp/cifar10/test')

    print("Training Starts")

    mon_sess = tf.train.MonitoredTrainingSession(
              hooks=[tf.train.StopAtStepHook(last_step=FLAGS.max_steps),
                      tf.train.NanTensorHook(loss)
                      ],
              config=tf.ConfigProto(
                    log_device_placement=FLAGS.log_device_placement))
    step = -1
    while not mon_sess.should_stop():
      step += 1
      _,loss_value = mon_sess.run([train_op,loss])
      if step % FLAGS.log_frequency == 0:
          tr_acc,summary = mon_sess.run([accuracy,merged], feed_dict = {train_flag : True})
          train_writer.add_summary(summary, step)
          te_acc, summary = mon_sess.run([accuracy, merged], feed_dict = {train_flag : False})
          test_writer.add_summary(summary, step)

          format_str = ('%s: step %d, loss = %.2f, test accuracy = %.2f, train accuracy = %.2f')
          print (format_str % (datetime.now(), step, loss_value, te_acc, tr_acc))


def main(argv=None):  
  cifar10.maybe_download_and_extract()
  if tf.gfile.Exists(FLAGS.train_dir):
      tf.gfile.DeleteRecursively(FLAGS.train_dir)
  tf.gfile.MakeDirs(FLAGS.train_dir)
  train()


if __name__ == '__main__':
  tf.app.run()

