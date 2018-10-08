import os
import glob
import numpy as np
import math
import hdf5_getters as getter
basedir = 'D:/tech/Workspaces/eclipse-workspace/PyTest/MillionSongs/millionsongsubset_full/MillionSongSubset/'
ext='.h5'

year_p = np.array([1926, 1927, 1929, 1930, 1934, 1935, 1936, 1940, 1947, 1950, 1953,
         1954, 1955, 1956, 1957, 1958, 1959, 1960, 1961, 1962, 1963, 1964,
         1965, 1966, 1967, 1968, 1969, 1970, 1971, 1972, 1973, 1974, 1975,
         1976, 1977, 1978, 1979, 1980, 1981, 1982, 1983, 1984, 1985, 1986,
         1987, 1988, 1989, 1990, 1991, 1992, 1993, 1994, 1995, 1996, 1997,
         1998, 1999, 2000, 2001, 2002, 2003, 2004, 2005, 2006, 2007, 2008,
         2009, 2010])


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
            except:
                print(1)
            h5.close()
    return years, timbres, pitches,min_length


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


'''
function: dataset_split
split traing_set=60%,cv_set=20%,test_set=20%
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

    
    
def transfer_year_to_68d(years):
    m = np.size(years)
    target_year = np.zeros([m,68])
    for i in range(m):
        y = years[i]
        tmp_y=year_p==y
        target_year[i,:] = tmp_y
    return target_year

def transfer_68d_to_year(years_68d):
    m = np.size(years_68d,0)
    year_indice = np.argmax(years_68d,1)
    target_year = year_p[year_indice]
    return target_year
