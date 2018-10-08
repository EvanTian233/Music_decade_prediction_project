import numpy as np
import RNN_Trained
import songProcess 
import pickle
import tensorflow as tf
import os
from sklearn.preprocessing import MinMaxScaler

os.environ["CUDA_VISIBLE_DEVICES"]="0"

iteration = 50000
batch_size = 64


file_name='warehouse'
fileObject = open(file_name,'rb')  
years = pickle.load(fileObject)
timbres = pickle.load(fileObject)
pitches = pickle.load(fileObject)
min_length = pickle.load(fileObject)
non_time_features = pickle.load(fileObject)
fileObject.close()

train_timbres,cv_timbres,test_timbres = songProcess.dataset_split(timbres)
train_pitches,cv_pitches,test_pitches = songProcess.dataset_split(pitches)
train_non_time,cv_non_time,test_non_time = songProcess.dataset_split(non_time_features)
train_year,cv_year,test_year = songProcess.dataset_split(years)

def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
    biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

xs = tf.placeholder(tf.float32,[None,20])
ys = tf.placeholder(tf.float32,[None,10])

l1 = add_layer(xs,20, 4, activation_function=tf.sigmoid)
#l2 = add_layer(l1, 64, 32, activation_function=tf.sigmoid)
#l3 = add_layer(l2, 512, 256, activation_function=tf.sigmoid)
#l4 = add_layer(l3, 256, 128, activation_function=tf.sigmoid)
prediction = add_layer(l1, 4, 10, activation_function = None)
loss = tf.losses.softmax_cross_entropy(onehot_labels=ys, logits=prediction)

train_step = tf.train.AdamOptimizer(0.01).minimize(loss)
#AdamOptimizer(0.01).minimize(loss)
#GradientDescentOptimizer(0.00000000005).minimize(loss)

accuracy = tf.metrics.accuracy(          # return (acc, update_op), and create 2 local variables
    labels=tf.argmax(ys, axis=1), predictions=tf.argmax(prediction, axis=1),)[1]

init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) 

sess = tf.Session()
sess.run(init_op)
tmp_d = RNN_Trained.getDecadesFromRNN(np.arange(0, np.size(train_year)),min_length,train_timbres, train_pitches,train_year)
global_decades = songProcess.transfer_year_to_10d(tmp_d) 
tmp_d2 = RNN_Trained.getDecadesFromRNN(np.arange(0, np.size(cv_year)),min_length,cv_timbres, cv_pitches,cv_year) 
global_cv_decades = songProcess.transfer_year_to_10d(tmp_d2)
tmp_d3 = RNN_Trained.getDecadesFromRNN(np.arange(0, np.size(test_year)),min_length,test_timbres, test_pitches,test_year) 
global_test_decades = songProcess.transfer_year_to_10d(tmp_d3)

index_array = np.arange(0, np.size(train_year)) 
train_X_without_year,train_y = songProcess.get_batch_non_time_data(index_array,train_non_time,train_year)
train_year10 = RNN_Trained.get_batch_decades_from_RNN(index_array,global_decades) 
train_X = np.hstack([train_X_without_year,train_year10])
scaler = MinMaxScaler(feature_range=(-5,5))
scaler.fit(train_X)

cv_array = np.arange(0, len(cv_year))
cv_X_without_year,cv_y = songProcess.get_batch_non_time_data(cv_array,cv_non_time,cv_year)
cv_year10 = RNN_Trained.get_batch_decades_from_RNN(cv_array,global_cv_decades) 
cv_X = np.hstack([cv_X_without_year,cv_year10])
cv_X = scaler.transform(cv_X)
cv_y = songProcess.transfer_year_to_decade(cv_y)

test_array = np.arange(0, len(test_year))
test_X_without_year,test_y = songProcess.get_batch_non_time_data(test_array,test_non_time,test_year)
test_year10 = RNN_Trained.get_batch_decades_from_RNN(test_array,global_test_decades) 
test_X = np.hstack([test_X_without_year,test_year10])
test_X = scaler.transform(test_X)
test_y = songProcess.transfer_year_to_decade(test_y)

for i in range(iteration):
    
    np.random.shuffle(index_array)
    batch_indices = index_array[0:batch_size]
    year_10 = RNN_Trained.get_batch_decades_from_RNN(batch_indices,global_decades) 
    batch_X_without_year,batch_y = songProcess.get_batch_non_time_data(batch_indices,train_non_time,train_year)
    batch_X = np.hstack([batch_X_without_year,year_10])
    #sess.run(train_step, feed_dict={xs: batch_X, ys: songProcess.transfer_year_to_68d(batch_y)})
    batch_X = scaler.transform(batch_X)
    batch_y = songProcess.transfer_year_to_decade(batch_y)
    d_, loss_ = sess.run([train_step, loss], {xs: batch_X, ys: songProcess.transfer_year_to_10d(batch_y)})
    if i%50==0:
        accuracy_ = sess.run(accuracy, {xs: batch_X, ys:  songProcess.transfer_year_to_10d(batch_y)})
        print(str(i)+'train loss: %.4f' % loss_, '| train accuracy: %.2f'% accuracy_)
accuracy_cv = sess.run(accuracy, feed_dict={xs: cv_X, ys:  songProcess.transfer_year_to_10d(cv_y)})
print('| cv accuracy: %.2f' % accuracy_cv)
accuracy_test = sess.run(accuracy, feed_dict={xs: test_X, ys:  songProcess.transfer_year_to_10d(test_y)})
print('| test accuracy: %.2f' % accuracy_test)

