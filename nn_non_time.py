import numpy as np
import songProcess 
import pickle
import tensorflow as tf
import os
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import MinMaxScaler

os.environ["CUDA_VISIBLE_DEVICES"]="0"

iteration = 15000
batch_size = 1000


file_name='warehouse'
fileObject = open(file_name,'rb')  
years = pickle.load(fileObject)
timbres = pickle.load(fileObject)
pitches = pickle.load(fileObject)
min_length = pickle.load(fileObject)
non_time_features = pickle.load(fileObject)
fileObject.close()

#train_timbres,cv_timbres,test_timbres = songProcess.dataset_split(timbres)
#train_pitches,cv_pitches,test_pitches = songProcess.dataset_split(pitches)
train_non_time,cv_non_time,test_non_time = songProcess.dataset_split(non_time_features)
train_year,cv_year,test_year = songProcess.dataset_split(years)

#sklearn-begin
'''
cv_array = np.arange(0, len(cv_year))
cv_X_without_year,cv_y = songProcess.get_batch_non_time_data(cv_array,cv_non_time,cv_year)
cv_y = songProcess.transfer_year_to_decade(cv_y)

index_array = np.arange(0, np.size(train_year)) 
batch_X_without_year,batch_y = songProcess.get_batch_non_time_data(index_array,train_non_time,train_year)
batch_y_10d = songProcess.transfer_year_to_10d(songProcess.transfer_year_to_decade(batch_y))

scaler = MinMaxScaler(feature_range=(-5,5))
scaler.fit(batch_X_without_year)
batch_X_without_year = scaler.transform(batch_X_without_year)
cv_X_without_year = scaler.transform(cv_X_without_year)

clf = MLPClassifier(solver='adam',hidden_layer_sizes=(64,32), learning_rate_init=0.00001,
                    max_iter=10000, momentum=0.9,batch_size=200,activation='relu')
clf.fit(batch_X_without_year, batch_y_10d)
result = clf.predict_proba(batch_X_without_year)
predict_decades = songProcess.transfer_10d_to_year(result)
acc = np.sum(predict_decades==batch_y)/np.size(batch_y)
print(acc)
#sklern-finish
'''
def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.random_normal([in_size, out_size]))
   # biases = tf.Variable(tf.zeros([1, out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) #+ biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs

xs = tf.placeholder(tf.float32,[None,10])
ys = tf.placeholder(tf.float32,[None,10])

l1 = add_layer(xs,10, 512, activation_function=tf.sigmoid)
l2 = add_layer(l1, 512, 256, activation_function=tf.sigmoid)
l3 = add_layer(l2, 256, 166, activation_function=tf.sigmoid)
l4 = add_layer(l3, 166, 90, activation_function=tf.sigmoid)

prediction = add_layer(l4, 90, 10, activation_function = None)
loss = tf.losses.softmax_cross_entropy(onehot_labels=ys, logits=prediction)

train_step = tf.train.AdamOptimizer(0.001).minimize(loss)
#AdamOptimizer(0.01).minimize(loss)
#GradientDescentOptimizer(0.00000000005).minimize(loss)
accuracy = tf.metrics.accuracy(          # return (acc, update_op), and create 2 local variables
    labels=tf.argmax(ys, axis=1), predictions=tf.argmax(prediction, axis=1),)[1]

init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer()) 

sess = tf.Session()
sess.run(init_op)

cv_array = np.arange(0, len(cv_year))
cv_X_without_year,cv_y = songProcess.get_batch_non_time_data(cv_array,cv_non_time,cv_year)
cv_y = songProcess.transfer_year_to_decade(cv_y)


for i in range(iteration):
    index_array = np.arange(0, np.size(train_year)) 
    np.random.shuffle(index_array)
    batch_indices = index_array[0:batch_size]
    batch_X_without_year,batch_y = songProcess.get_batch_non_time_data(batch_indices,train_non_time,train_year)
    batch_decade = songProcess.transfer_year_to_decade(batch_y)
    one_hot_decade = songProcess.transfer_year_to_10d(batch_decade)
    sess.run(train_step,feed_dict={xs: batch_X_without_year, ys: one_hot_decade})
    if i%50==0:
        loss_ = sess.run(loss, {xs: batch_X_without_year, ys: one_hot_decade})
        accuracy_ = sess.run(accuracy, {xs: batch_X_without_year, ys:  one_hot_decade})
        print(str(i)+'train loss: %.4f' % loss_, '| train accuracy: %.2f'% accuracy_)
prediction = sess.run(prediction, feed_dict={xs: cv_X_without_year, ys:  songProcess.transfer_year_to_10d(cv_y)})
predict_decades = songProcess.transfer_10d_to_year(prediction)
acc = np.sum(predict_decades==cv_y)/np.size(cv_y)
print(acc)
#print('| cv accuracy: %.2f' % accuracy_cv)
