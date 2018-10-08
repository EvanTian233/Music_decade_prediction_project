'''
Created on March 15th,2018

@author: Rhys Wang
'''
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#hyperparameters
lr = 0.0001
iterations = 100000
batch_size = 128

n_inputs = 28 #input nodes
n_steps = 28 #time steps
n_hidden_units = 128
n_classes = 10

# x y placeholder
x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

#define weights
weights = {
    # shape (28, 128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # shape (128, 10)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}

#define bias
biases = {
    # shape (128, )
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # shape (10, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}

def RNN(X,weights,biases):
    #hidden layer from input to cell
    #X.shape = batch_size*n_steps*n_inputs
    #==>batch_size*n_steps,n_inputs
    X = tf.reshape(X, [-1,n_inputs])
    #X_in.shape = [batch_size*n_steps,n_hidden_units]
    X_in = tf.matmul(X,weights['in'])+biases['in']
    #X_in.shape = [batch_size,n_steps,n_hidden_units]
    X_in = tf.reshape(X_in,[-1,n_steps,n_hidden_units])
    
    #cell
    lstm_cell = tf.contrib.rnn.BasicLSTMCell(n_hidden_units, forget_bias=1.0, state_is_tuple=True)
    #ltms state is divided into two states:c_state and m_state
    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32) 
    outputs, final_state = tf.nn.dynamic_rnn(lstm_cell, X_in, initial_state=init_state, time_major=False)
    #hidden layer from cell to output
    outputs = tf.unstack(tf.transpose(outputs, [1,0,2]))
    results = tf.matmul(outputs[-1], weights['out']) + biases['out']
    return results

pred = RNN(x, weights, biases)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = y,logits = pred))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)
correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))

init = tf.global_variables_initializer()

mnist=input_data.read_data_sets('MNIST_data',one_hot=True)

with tf.Session() as sess:
    sess.run(init)
    step = 0
    while step * batch_size < iterations:
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        sess.run([train_op], feed_dict={
            x: batch_xs,
            y: batch_ys,
        })
        if step % 20 == 0:
            print(sess.run(accuracy, feed_dict={
            x: batch_xs,
            y: batch_ys,
        }))
        step += 1
