'''
Feb 26th,2018
@author: rhys
'''
import os
import glob
import numpy as np
from MillionSongs import hdf5_getters
from MillionSongs import display_song

basedir = 'D:/tech/Workspaces/eclipse-workspace/PyTest/MillionSongs/millionsongsubset_full/MillionSongSubset'
ext='.h5'

def count_all_files() :
    cnt = 0
    for root, dirs, files in os.walk(basedir):
        files = glob.glob(os.path.join(root,'*'+ext))
        cnt += len(files)
        print(cnt)
    return cnt
