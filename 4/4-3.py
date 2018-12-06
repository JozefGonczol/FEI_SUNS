from __future__ import print_function
import numpy as np
import tensorflow as tf
import os.path
from six.moves import cPickle as pickle
from six.moves import range

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

pickle_file = '../data/all_data.pickle'

with open(pickle_file, 'rb') as f:
    save = pickle.load(f)
    train_dataset = save['train_data']
    train_labels = save['train_labels']
    valid_dataset = save['valid_data']
    valid_labels = save['valid_labels']
    test_dataset = save['test_data']
    test_labels = save['test_labels']
    del save  # hint to help gc free up memory
    print('Training set', train_dataset.shape, train_labels.shape)
    print('Validation set', valid_dataset.shape, valid_labels.shape)
    print('Test set', test_dataset.shape, test_labels.shape)

image_size = 100
num_labels = 48


def reformat(dataset, labels, dict):
    size = len(dataset[0])
    dataset = dataset.reshape((-1, size * size * 3)).astype(np.float32)
    # Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
    labels = transform_labesl(labels, dict)
    labels = (np.arange(num_labels) == labels[:, None]).astype(np.float32)
    return dataset, labels


def create_dict():
    dict = {}
    buff = 1
    for dir in os.listdir("../data/Test"):
        if os.path.isfile("../data/Test/{}".format(dir)):
            name = dir.split(".")[0]
            dict[name] = buff
            buff += 1
        else:
            continue
    return dict


def transform_labesl(labels, dicto):
    newlabels = []
    for label in labels:
        lbl = dicto[label]
        if len(newlabels) is 0:
            newlabels = [lbl]
        else:
            newlabels.extend([lbl])
    return np.asarray(newlabels)


dicto = create_dict()
train_dataset, train_labels = reformat(train_dataset, train_labels, dicto)
valid_dataset, valid_labels = reformat(valid_dataset, valid_labels, dicto)
test_dataset, test_labels = reformat(test_dataset, test_labels, dicto)
print('Training set', train_dataset.shape, train_labels.shape, dict)
print('Validation set', valid_dataset.shape, valid_labels.shape, dict)
print('Test set', test_dataset.shape, test_labels.shape)

# With gradient descent training, even this much data is prohibitive.
# Subset the training data for faster turnaround.
train_subset = 10000


def accuracy(predictions, labels):
    return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
            / predictions.shape[0])


batch_size = 128
hidden_layer = 1024
ovetrian_batches = 4

graph = tf.Graph()
with graph.as_default():
    # Input data. For the training data, we use a placeholder that will be fed
    # at run time with a training minibatch.
    tf_train_dataset = tf.placeholder(tf.float32,
                                      shape=(batch_size, image_size * image_size * 3))
    tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
    tf_valid_dataset = tf.constant(valid_dataset)
    tf_test_dataset = tf.constant(test_dataset)
    l2Regular = tf.placeholder(tf.float32)

    # in
    in_weights = tf.Variable(
        tf.truncated_normal([image_size * image_size * 3, hidden_layer]))
    in_biases = tf.Variable(tf.zeros([hidden_layer]))
    in_train = tf.nn.relu(tf.matmul(tf_train_dataset, in_weights) + in_biases)
    # out
    out_weights = tf.Variable(tf.truncated_normal([hidden_layer, num_labels]))
    out_biases = tf.Variable(tf.zeros([num_labels]))

    # Training computation.
    logits = tf.matmul(in_train, out_weights) + out_biases
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=tf_train_labels, logits=logits) + l2Regular * (
            tf.nn.l2_loss(in_weights) + tf.nn.l2_loss(in_weights)))

    # Optimizer.
    # optimizer = tf.train.GradientDescentOptimizer(0.5).minimize(loss)
    optimizer = tf.train.AdamOptimizer(0.5).minimize(loss)

# Predictions for the training, validation, and test data.
train_prediction = tf.nn.softmax(logits)
in_valid = tf.nn.relu(tf.matmul(tf_valid_dataset, in_weights) + in_biases)
valid_prediction = tf.nn.softmax(tf.matmul(in_valid, out_weights) + out_biases)
in_test = tf.nn.relu(tf.matmul(tf_test_dataset, in_weights) + in_biases)
test_prediction = tf.nn.softmax(tf.matmul(in_test, out_weights) + out_biases)


relu = 0.005

steps = [200, 300, 400, 500, 600]

with tf.Session(graph=graph) as session:
    tf.global_variables_initializer().run()
    print("Initialized")
    for num_steps in steps:
        for step in range(num_steps):
            # Pick an offset within the training data, which has been randomized.
            # Note: we could use better randomization across epochs.
            offset = (step * batch_size) % (train_labels.shape[0] - batch_size)
            # Generate a minibatch.
            batch_data = train_dataset[offset:(offset + batch_size), :]
            batch_labels = train_labels[offset:(offset + batch_size), :]
            # Prepare a dictionary telling the session where to feed the minibatch.
            # The key of the dictionary is the placeholder node of the graph to be fed,
            # and the value is the numpy array to feed to it.
            feed_dict = {tf_train_dataset: batch_data, tf_train_labels: batch_labels, l2Regular: relu}
            _, l, predictions = session.run(
                [optimizer, loss, train_prediction], feed_dict=feed_dict)
            if (step % 100 == 0):
                print("Minibatch loss at step %d: %f" % (step, l))
                print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
                print("Validation accuracy: %.1f%%" % accuracy(
                    valid_prediction.eval(), valid_labels))
        # print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))
