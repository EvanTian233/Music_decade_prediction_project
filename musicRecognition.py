import numpy as np
import songProcess_back 
import pickle

'''Hyper parameter'''

batch_size = 64

'''Data processing'''
#read data
file_name='warehouse1'
fileObject = open(file_name,'rb')  
years = pickle.load(fileObject)
timbres = pickle.load(fileObject)
pitches = pickle.load(fileObject)
min_length = pickle.load(fileObject)
fileObject.close()
total_size = len(years)

'''Data Spliting'''
#training set
train_timbres,cv_timbres,test_timbres = songProcess_back.dataset_split(timbres)
train_pitches,cv_pitches,test_pitches = songProcess_back.dataset_split(pitches)
train_years,cv_years,test_years = songProcess_back.dataset_split(years)

'''Get batch set'''
#get batch ids
index_array = np.arange(0,len(train_years))
np.random.shuffle(index_array)
batch_indices = index_array[0:batch_size]
#get batch_x and batch_y
batch_X,batch_y = songProcess_back.get_batch_data(batch_indices,min_length,train_timbres,train_pitches,train_years)
print(1)