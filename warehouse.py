import songProcess 
import pickle

file_name='warehouse'
years, timbres, pitches,min_length,ten_features = songProcess.load_raw_data()
fileObject = open(file_name,'wb')
pickle.dump(years,fileObject)
pickle.dump(timbres,fileObject)
pickle.dump(pitches,fileObject)
pickle.dump(min_length,fileObject)
pickle.dump(ten_features,fileObject)
fileObject.close()