# Copyright 2015 Google Inc. All Rights Reserved.
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

"""Trains and Evaluates the  network using a feed dictionary."""
# pylint: disable=missing-docstring
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os.path
import time

import numpy as np
from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import read_data
import mnist_tb

def next_batch(step,vectors_data,labels_data, batch_size, fake_data=False):
    """Return the next `batch_size` examples from this data set."""
    _index_in_epoch = step*batch_size
    _num_examples = vectors_data.shape[0]
    _epochs_completed = 0
    if fake_data:
      fake_vector = [1] * 40
      if self.one_hot:
        fake_label = [1] + [0] * 1
      else:
        fake_label = 0
      return [fake_vector for _ in xrange(batch_size)], [
          fake_label for _ in xrange(batch_size)]
    start = _index_in_epoch
    _index_in_epoch += batch_size
    if _index_in_epoch > _num_examples:
      # Finished epoch
      _epochs_completed += 1
      # Shuffle the data
      perm = np.arange(_num_examples)
      np.random.shuffle(perm)
      vectors = vectors_data[perm]
      labels = labels_data[perm]
      # Start next epoch
      start = 0
      _index_in_epoch = batch_size
      assert batch_size <= _num_examples
    end = _index_in_epoch

    return vectors_data[start:end], labels_data[start:end]

# Basic model parameters as external flags.
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_float('learning_rate', 0.01, 'Initial learning rate.')
flags.DEFINE_integer('max_steps', 2000, 'Number of steps to run trainer.')
flags.DEFINE_integer('hidden1', 28, 'Number of units in hidden layer 1.')
flags.DEFINE_integer('hidden2', 16, 'Number of units in hidden layer 2.')
flags.DEFINE_integer('batch_size', 100, 'Batch size.  '
                     'Must divide evenly into the dataset sizes.')
flags.DEFINE_string('train_dir', 'train/full3', 'Directory to put the training data.')
flags.DEFINE_boolean('fake_data', False, 'If true, uses fake data '
                     'for unit testing.')


def placeholder_inputs(batch_size):
  """Generate placeholder variables to represent the input tensors.
  These placeholders are used as inputs by the rest of the model building
  code and will be fed from the downloaded data in the .run() loop, below.
  Args:
    batch_size: The batch size will be baked into both placeholders.
  Returns:
    images_placeholder: Images placeholder.
    labels_placeholder: Labels placeholder.
  """
  # Note that the shapes of the placeholders match the shapes of the full
  # image and label tensors, except the first dimension is now batch_size
  # rather than the full size of the train or test data sets.
  vectors_placeholder = tf.placeholder(tf.float32, shape=(batch_size,
                                                         mnist_tb.VECTOR_SIZE))
  labels_placeholder = tf.placeholder(tf.int32, shape=(batch_size))
  return vectors_placeholder, labels_placeholder


def fill_feed_dict(step,vectors_data,labels_data, vectors_pl, labels_pl):
  """Fills the feed_dict for training the given step.
  A feed_dict takes the form of:
  feed_dict = {
      <placeholder>: <tensor of values to be passed for placeholder>,
      ....
  }
  Args:
    data_set: The set of images and labels, from input_data.read_data_sets()
    images_pl: The images placeholder, from placeholder_inputs().
    labels_pl: The labels placeholder, from placeholder_inputs().
  Returns:
    feed_dict: The feed dictionary mapping from placeholders to values.
  """
  # Create the feed_dict for the placeholders filled with the next
  # `batch size ` examples.
  vectors_feed, labels_feed = next_batch(step,vectors_data,labels_data,FLAGS.batch_size,
                                                 FLAGS.fake_data)
  feed_dict = {
      vectors_pl: vectors_feed,
      labels_pl: labels_feed,
  }
  return feed_dict


def do_eval(sess,
            eval_correct,
            vectors_placeholder,
            labels_placeholder,
            vectors_data,
            labels_data):
  """Runs one evaluation against the full epoch of data.
  Args:
    sess: The session in which the model has been trained.
    eval_correct: The Tensor that returns the number of correct predictions.
    images_placeholder: The images placeholder.
    labels_placeholder: The labels placeholder.
    data_set: The set of images and labels to evaluate, from
      input_data.read_data_sets().
  """
  # And run one epoch of eval.
  true_count = 0  # Counts the number of correct predictions.
  steps_per_epoch = vectors_data.shape[0] // FLAGS.batch_size
  num_examples = steps_per_epoch * FLAGS.batch_size

  with tf.name_scope("evaluation"):
    for step in xrange(steps_per_epoch):
      feed_dict = fill_feed_dict(step,vectors_data,labels_data,
                               vectors_placeholder,
                               labels_placeholder)
      true_count += sess.run(eval_correct, feed_dict=feed_dict)
  
  precision = true_count / num_examples

  print('  Num examples: %d  Num correct: %d  Precision @ 1: %0.04f' %
        (num_examples, true_count, precision))


def run_training():
  """Train  for a number of steps."""
  # Get the sets of images and labels for training, validation, and
  # test on .
  filename = "Data11-17.txt"
  vectors_data1,labels_data1 = read_data.read_data(filename)
  filename = "valid18-20.txt"
  vectors_data2,labels_data2 = read_data.read_data(filename)
  filename = "Data21-25.txt"
  vectors_data3,labels_data3 = read_data.read_data(filename)

  vectors_data = np.vstack((vectors_data1,vectors_data2,vectors_data3))
  print(vectors_data.shape)
  labels_data = np.vstack((np.reshape(labels_data1,(len(labels_data1),1)),
    np.reshape(labels_data2,(len(labels_data2),1)),np.reshape(labels_data3,(len(labels_data3),1))))
  labels_data = np.reshape(labels_data,-1)
  print(labels_data.shape)

  filename = "Data4-10.txt"
  validation_data,vlabels_data = read_data.read_data(filename)
  filename = "Data26-29.txt"
  test_data,tlabels_data = read_data.read_data(filename)


  # Tell TensorFlow that the model will be built into the default Graph.
  with tf.Graph().as_default() as data:
    # Generate placeholders for the images and labels.
    vectors_placeholder, labels_placeholder = placeholder_inputs(
        FLAGS.batch_size)

    # Build a Graph that computes predictions from the inference model.
    logits = mnist_tb.inference(vectors_placeholder,
                             FLAGS.hidden1,
                             FLAGS.hidden2)

    # Add to the Graph the Ops for loss calculation.
    loss = mnist_tb.loss(logits, labels_placeholder)

    # Add to the Graph the Ops that calculate and apply gradients.
    train_op = mnist_tb.training(loss, FLAGS.learning_rate)

    # Add the Op to compare the logits to the labels during evaluation.
    eval_correct = mnist_tb.evaluation(logits, labels_placeholder)

    # Build the summary operation based on the TF collection of Summaries.
    summary_op = tf.merge_all_summaries()

    # Create a saver for writing training checkpoints.
    saver = tf.train.Saver()

    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    # Run the Op to initialize the variables.
    init = tf.initialize_all_variables()
    sess.run(init)

    # Instantiate a SummaryWriter to output summaries and the Graph.

    summary_writer = tf.train.SummaryWriter(FLAGS.train_dir,sess.graph)


    # And then after everything is built, start the training loop.
    for step in xrange(FLAGS.max_steps):
      start_time = time.time()

      # Fill a feed dictionary with the actual set of images and labels
      # for this particular training step.
      feed_dict = fill_feed_dict(step,vectors_data,labels_data,
                                 vectors_placeholder,
                                 labels_placeholder)

      # Run one step of the model.  The return values are the activations
      # from the `train_op` (which is discarded) and the `loss` Op.  To
      # inspect the values of your Ops or variables, you may include them
      # in the list passed to sess.run() and the value tensors will be
      # returned in the tuple from the call.
      _, loss_value = sess.run([train_op, loss],
                               feed_dict=feed_dict)

      duration = time.time() - start_time

      # Write the summaries and print an overview fairly often.
      if step % 100 == 0:
        # Print status to stdout.
        print('Step %d: loss = %.2f (%.3f sec)' % (step, loss_value, duration))
        # Update the events file.
        summary_str = sess.run(summary_op, feed_dict=feed_dict)
        summary_writer.add_summary(summary_str, step)

        saver.save(sess, FLAGS.train_dir, global_step=step)

      # Save a checkpoint and evaluate the model periodically.
      if (step + 1) % 1000 == 0 or (step + 1) == FLAGS.max_steps:
        saver.save(sess, FLAGS.train_dir, global_step=step)
        # Evaluate against the training set.
        print('Training Data Eval:')
        do_eval(sess,
                eval_correct,
                vectors_placeholder,
                labels_placeholder,
                vectors_data,
                labels_data)
        # Evaluate against the validation set.
        print('Validation Data Eval:')
        do_eval(sess,
                eval_correct,
                vectors_placeholder,
                labels_placeholder,
                validation_data,
                vlabels_data)
        # Evaluate against the test set.
        print('Test Data Eval:')
        do_eval(sess,
                eval_correct,
                vectors_placeholder,
                labels_placeholder,
                test_data,
                tlabels_data)


def main(_):
  run_training()


if __name__ == '__main__':
  tf.app.run()