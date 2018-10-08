import tensorflow as tf
import numpy as np
import songProcess_back
import pickle
import os
import songProcess
os.environ["CUDA_VISIBLE_DEVICES"]="0"

tf.set_random_seed(1) 
np.random.seed(1)

'''Hyper parameter'''
# Hyper Parameters
INPUT_SIZE = 24         # rnn input size
batch_size = 64

'''Data processing'''
#read data
file_name='warehouse'
fileObject = open(file_name,'rb')  
years = pickle.load(fileObject)
timbres = pickle.load(fileObject)
pitches = pickle.load(fileObject)
min_length = pickle.load(fileObject)
fileObject.close()

'''RNN Model Definition'''
# tensorflow placeholders
tf_x = tf.placeholder(tf.float32, [None, min_length * INPUT_SIZE])       # shape(batch, 784)   [time_steps, batch_size, num_features]
image = tf.reshape(tf_x, [-1, INPUT_SIZE,min_length])                   # (batch, height, width, channel)
tf_y = tf.placeholder(tf.int32, [None, 10])                             # input y

# RNN
rnn_cell = tf.contrib.rnn.BasicLSTMCell(num_units=300)
outputs, (h_c, h_n) = tf.nn.dynamic_rnn(
    rnn_cell,                   # cell you have chosen
    image,                      # input
    initial_state=None,         # the initial hidden state
    dtype=tf.float32,           # must given if set initial_state = None
    time_major=False,           # False: (batch, time step, input); True: (time step, batch, input)
)
output = tf.layers.dense(outputs[:, -1, :], 10)              # output based on the last output step
saver = tf.train.Saver()

def getDecadesFromRNN(indices,min_length, timbres, pitches,years):
    batch_X, batch_y = songProcess.get_batch_data(indices, min_length, timbres, pitches, songProcess.transfer_year_to_decade(years))
    m = np.size(indices)
    logits = np.zeros([m,10])
    '''Restore RNN Session'''
    with tf.Session() as sesssion:
        #sesssion.run(init_lo)
        saver.restore(sesssion, "my_net/save_net_rnn.ckpt")
        # print 10 predictions from test data
        logits = sesssion.run(output, feed_dict={tf_x: batch_X, tf_y:  songProcess.transfer_year_to_10d(batch_y)})
        decades = songProcess.transfer_10d_to_year(logits)
        sesssion.close()
    return decades

def soft_max(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()

def getDecadesProbFromRNN(indices,min_length, timbres, pitches,years):
    batch_X, batch_y = songProcess.get_batch_data(indices, min_length, timbres, pitches, songProcess.transfer_year_to_decade(years))
    m = np.size(indices)
    logits = np.zeros([m,10])
    '''Restore RNN Session'''
    with tf.Session() as sesssion:
        #sesssion.run(init_lo)
        saver.restore(sesssion, "my_net/save_net_rnn.ckpt")
        # print 10 predictions from test data
        logits = sesssion.run(output, feed_dict={tf_x: batch_X, tf_y:  songProcess.transfer_year_to_10d(batch_y)})
        decades = soft_max(logits)
        sesssion.close()
    return decades

def get_batch_decades_from_RNN(indices,global_10d_decades):
    size = np.size(indices)
    logits = np.zeros([size,10])
    for i in range(0,size):
        id = indices[i]
        year10 = global_10d_decades[id,:] 
        logits[i] = year10
    return logits
