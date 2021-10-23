# -*- coding: utf-8 -*-
"""
Created on Thu Oct 21 18:13:08 2021

@author: ramim
"""

filenames = ['Data\cross-era_chroma-nnls\chroma-nnls_orchestra_romantic.csv',
             'Data\cross-era_chroma-nnls\chroma-nnls_orchestra_classical.csv',
             'Data\cross-era_chroma-nnls\chroma-nnls_orchestra_baroque.csv',
             'Data\cross-era_chroma-nnls\chroma-nnls_orchestra_modern.csv',
             'Data\cross-era_chroma-nnls\chroma-nnls_orchestra_addon.csv']

import pandas as pd
import numpy as np
import random

data_arrays = []
for filename in filenames:
    # Read the data through pandas from our csv file
    data = pd.read_csv(filename, sep = ',', skiprows = 0, header=None)
    
    # Turn our pandas data to a numpy 2d array
    data = data.to_numpy()
    
    # Find row indices for the start of songs
    indices = []
    cur = 0
    
    for row in data:
        if(pd.isna(row[0]) is False):
            indices.append(cur)
        cur += 1
        
    # Add songs to a dictionary that maps names to the respective 2d np array, we chop the nan column as well
    songs = {}
    songNames = []
    
    for i in range(len(indices)-1):
        song = data[indices[i]+1:indices[i+1],2:].copy()
        songName = data[indices[i],0]
        songs[songName] = song
        songNames.append(songName)
    
    # Choose a cutoff time (use the commented out code to check)
    cutoff = 750
    
    # Delete songs shorter than the cutoff time
    for songKey in list(songs.keys()):
        song = songs[songKey]
        if (song.shape[0] < cutoff):
            songs.pop(songKey)
    max = 0
    # Trim all other songs to cutoff time
    for key in songs.keys():
        songs[key] = songs[key][:cutoff,:].flatten().astype(float)
        if(np.amax(songs[key]) > max):
            max = np.amax(songs[key])
    print(max)
    
    
    data_arrays.append(np.array(list(songs.values()))) 
    print(np.array(list(songs.values())).shape)
    
np.savez('orchestra_data.npz', romantic=data_arrays[0],classical=data_arrays[1],baroque=data_arrays[2],modern=data_arrays[3],addon=data_arrays[4])



"""
runtimes = 0
cutSongs = 0
runtimecutoff = 750
minruntime = random.choice(list(songs.values())).shape[0]

for song in songs.values():
    if(song.shape[0] < minruntime):
        minruntime = song.shape[0]
    if(song.shape[0] < runtimecutoff):
        cutSongs += 1
    runtimes += song.shape[0]

print("There are {0} songs in this dataset".format(len(songs.keys())))
print("There are {0} songs shorter than {1} seconds in this dataset".format(cutSongs, runtimecutoff))
    
"""    
## We choose a tolerance and turn the note on or off based on that tolerance
# Song in songs.values():
#    Song = (song > 0.5).astype(int)

    