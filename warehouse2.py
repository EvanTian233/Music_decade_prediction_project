import songProcess 
import pickle

file_name='warehouse2'
years,ten_features = songProcess.load_non_time_data()
fileObject = open(file_name,'wb')
pickle.dump(years,fileObject)
pickle.dump(ten_features,fileObject)
fileObject.close()