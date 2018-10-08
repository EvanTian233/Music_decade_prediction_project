import tensorflow as tf
import numpy as np
import songProcess
import pickle
import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"

tf.set_random_seed(1) 
np.random.seed(1)

'''Hyper parameter'''
# Hyper Parameters
INPUT_SIZE = 24         # rnn input size
LR = 0.0003               # learning rate
accuracy_cv = 0
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
total_size = len(years)

'''Data Spliting'''
#training set
train_timbres,cv_timbres,test_timbres = songProcess.dataset_split(timbres)
train_pitches,cv_pitches,test_pitches = songProcess.dataset_split(pitches)
train_years,cv_years,test_years = songProcess.dataset_split(years)
cv_array = np.arange(0, len(cv_years))
cv_X, cv_y = songProcess.get_batch_data(cv_array, min_length, cv_timbres, cv_pitches, cv_years)

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

loss = tf.losses.softmax_cross_entropy(onehot_labels=tf_y, logits=output)           # compute cost
train_op = tf.train.AdamOptimizer(LR).minimize(loss)

accuracy = tf.metrics.accuracy(          # return (acc, update_op), and create 2 local variables
    labels=tf.argmax(tf_y, axis=1), predictions=tf.argmax(output, axis=1),)[1]
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) 
saver = tf.train.Saver()

'''RNN main controler'''
sess = tf.Session()
sess.run(init_op)     # initialize var in graph
for step in range(5000):    # training iteration
    '''Get batch set'''
    # get batch ids
    index_array = np.arange(0, len(train_years))
    np.random.shuffle(index_array)
    batch_indices = index_array[0:batch_size]
    # get batch_x and batch_y
    batch_X, batch_y = songProcess.get_batch_data(batch_indices, min_length, train_timbres, train_pitches, songProcess.transfer_year_to_decade(train_years))
    _, loss_ = sess.run([train_op, loss], {tf_x: batch_X, tf_y: songProcess.transfer_year_to_10d(batch_y)})
    if step % 50 == 0:      # testing
        accuracy_ = sess.run(accuracy, {tf_x: batch_X, tf_y:  songProcess.transfer_year_to_10d(batch_y)})
        print('train loss: %.4f' % loss_, '| train accuracy: %.2f' % accuracy_)
        if accuracy_>0.985:
            save_path = saver.save(sess, "my_net/save_net_rnn.ckpt")
        #    accuracy_cv = sess.run(accuracy, feed_dict={tf_x: cv_X, tf_y:  songProcess_back.transfer_year_to_68d(cv_y)})
        #    print('cv accuracy: %.2f' % accuracy_cv)
            break;
accuracy_cv = sess.run(accuracy, feed_dict={tf_x: cv_X, tf_y:  songProcess.transfer_year_to_10d(cv_y)})
print('cv accuracy: %.2f' % accuracy_cv)
sess.close()
with tf.Session() as sesssion:
    #sesssion.run(init_lo)
    saver.restore(sesssion, "my_net/save_net_rnn.ckpt")
    # print 10 predictions from test data
    logits = sesssion.run(output, feed_dict={tf_x: cv_X, tf_y:  songProcess.transfer_year_to_10d(cv_y)})
    print(songProcess.transfer_10d_to_year(logits)) 
    sesssion.close()
#test_output = sess.run(output, {tf_x: cv_X})
#pred_y = np.argmax(test_output, 1)
#print(pred_y, 'prediction number')
#print(np.argmax(cv_y, 1), 'real number') 