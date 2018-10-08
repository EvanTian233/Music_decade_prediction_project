import os
import glob
import numpy as np
import math
import hdf5_getters as getter
from multiprocessing.connection import families
from hdf5_getters import get_end_of_fade_in
basedir =  'D:/tech/Workspaces/eclipse-workspace/PyTest/MillionSongs/millionsongsubset_full/MillionSongSubset/'
ext='.h5'

year_p = np.array([1920,1930,1940,1950,1960,1970,1980,1990,2000,2010])

'''
function: load_raw_data
load h5 files, and get two audio features, and the year data
return three list timbres,pitches,years
get rid of the data without year data or sample points less than 100
also return the min length, which is the least sample points in the set
no cut here
'''
def load_raw_data():
    years = []
    ten_features=[]
    timbres = []
    pitches = []
    min_length = 10000
    num = 0
    for root, dirs, files in os.walk(basedir):
        files = glob.glob(os.path.join(root,'*'+ext))
        for f in files:
            h5 = getter.open_h5_file_read(f)
            num += 1
            print(num)
            try:
                year = getter.get_year(h5)
                if year!=0:
                    timbre = getter.get_segments_timbre(h5)
                    s = np.size(timbre,0)
                    if s>=100:
                        if s<min_length:
                            min_length = s
                        pitch = getter.get_segments_pitches(h5)
                        years.append(year)
                        timbres.append(timbre)
                        pitches.append(pitch)
                        title_length = len(getter.get_title(h5))
                        terms_length = len(getter.get_artist_terms(h5))
                        tags_length = len(getter.get_artist_mbtags(h5))
                        hotness = getter.get_artist_hotttnesss(h5)
                        duration = getter.get_duration(h5)
                        loudness = getter.get_loudness(h5)
                        mode = getter.get_mode(h5)
                        release_length = len(getter.get_release(h5))
                        tempo = getter.get_tempo(h5)
                        name_length = len(getter.get_artist_name(h5))
                        ten_feature = np.hstack([title_length, hotness, duration, tags_length,
                                                 terms_length,loudness, mode, release_length, tempo, name_length])

                        ten_features.append(ten_feature) 
            except:
                print(1)
            h5.close()
    return years, timbres, pitches,min_length,ten_features

def load_non_time_data():
    years = []
    ten_features=[]
    num = 0
    for root, dirs, files in os.walk(basedir):
        files = glob.glob(os.path.join(root,'*'+ext))
        for f in files:
            h5 = getter.open_h5_file_read(f)
            num += 1
            print(num)
            try:
                year = getter.get_year(h5)
                if year!=0:
                    years.append(year)
                    title_length = len(getter.get_title(h5))
                    terms_length = len(getter.get_artist_terms(h5))
                    tags_length = len(getter.get_artist_mbtags(h5))
                    hotness = getter.get_artist_hotttnesss(h5)
                    duration = getter.get_duration(h5)
                    loudness = getter.get_loudness(h5)
                    mode = getter.get_mode(h5)
                    release_length = len(getter.get_release(h5))
                    tempo = getter.get_tempo(h5)
                    name_length = len(getter.get_artist_name(h5))
                    ten_feature = np.hstack([title_length,tags_length, hotness, duration,
                                             terms_length, loudness, mode, release_length, tempo, name_length])
                    ten_features.append(ten_feature) 
            except:
                print(1)
            h5.close()
    return years,ten_features





'''
function: get_batch_data
Given batch_size,batch_ids, minimum sample points, and the three data lists
return the batch_X and batch_y
batch_X looks like:
[
    batch_0: time0_timbre0,...,time0_timbre11,time0_pitch0,...,time0_pitch11,time2......time_min_length
    .
    .
    .
    batch_n:
]

'''
def get_batch_data(indices,min_length,timbres,pitches,years):
    
    batch_size = np.size(indices)
    #the data should be in the shape of[bath_size,24*min_length]
    batch_X = np.zeros([batch_size,24*min_length])
    batch_y = np.zeros([batch_size])
    #run over the batches
    for i in range(np.size(indices)):
        #get the song id
        id = indices[i]
        #attention:We cut the data here!!!!
        #get the timbre with time_step = min_length
        timbre = timbres[id][0:min_length]
        #get the pitch with time_step = min_length
        pitch = pitches[id][0:min_length]
        #run over the time step
        for j in range(min_length):
            #get the timbre and pitch in a specific time point
            timbre_jth_time = timbre[j,:].T
            pitch_j_th_time = pitch[j,:].T
            #put them together into the batch_X
            batch_X[i,j*24:(j+1)*24] = np.hstack([timbre_jth_time,pitch_j_th_time])
        #get year for the song 
        year = years[id]
        batch_y[i] = year
    return batch_X,batch_y


def get_batch_non_time_data(indices,non_time_features,years):
    
    batch_size = np.size(indices)
    num_of_features = np.size(non_time_features,1)
    batch_X = np.zeros([batch_size,num_of_features])
    batch_y = np.zeros([batch_size])
    #run over the batches
    for i in range(np.size(indices)):
        #get the song id
        id = indices[i]
        feature = non_time_features[id]
        batch_X[i,:] = feature
        year = years[id]
        batch_y[i] = year
    return batch_X,batch_y

'''
function: dataset_split
split traing_set=80%,cv_set=10%,test_set=10%
'''
def dataset_split(total_set):
    total_size = len(total_set)
    #Calculate size for the three sets
    training_size = math.floor(total_size*0.6)
    cv_size = (total_size-training_size)//2
    #test_size = total_size-training_size-cv_size
    
    #get everything from the list
    traing_set = total_set[0:training_size]
    cv_set = total_set[training_size:training_size+cv_size]
    test_set = total_set[training_size+cv_size:total_size]
    return traing_set,cv_set,test_set
    
    
def transfer_year_to_10d(years):
    m = np.size(years)
    target_year = np.zeros([m,10])
    for i in range(m):
        y = years[i]
        tmp_y=year_p==y
        target_year[i,:] = tmp_y
    return target_year

def transfer_10d_to_year(years_10d):
    m = np.size(years_10d,0)
    year_indice = np.argmax(years_10d,1)
    target_year = year_p[year_indice]
    return target_year

def transfer_year_to_decade(years):
    m = np.size(years)
    decades = np.zeros([m])
    for i in range(m):
        year = years[i]
        decade = (year//10)*10
        decades[i]=decade
    return decades
