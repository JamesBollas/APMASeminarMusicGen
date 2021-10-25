# -*- coding: utf-8 -*-
"""
Created on Fri Oct 22 14:16:26 2021

@author: ramim
"""

import numpy as np

import mingus.core.notes as notes
import mingus.core.scales as scales
import mingus.core.meter as meter
from mingus.containers import Note
from mingus.containers import NoteContainer
from mingus.containers import Bar
from mingus.containers import Track
from mingus.containers.instrument import Instrument
from mingus.containers.instrument import MidiInstrument
from mingus.containers import Composition
from mingus.midi import midi_file_out
import random

data_arrays = np.load('gan_out.npz')
#data = [data_arrays['classical'],data_arrays['baroque'],data_arrays['modern'],data_arrays['romantic'],data_arrays['addon']]
print(data_arrays['arr_0'][2])

        
notes_list = ['A', 'A#', 'B', 'C', 'C#', 'D', 'D#', 'E', 'F', 'F#', 'G', 'G#']

temp = data_arrays['arr_0'][3][:1500].reshape(-1,12)
#print(temp)
temp *= 5 / np.max(temp)
print(np.max(temp))
datas = [temp]
MAX_VELOCITY = 127
print(np.mean(datas[0]))
# threshold = 2.4
def choose_octave(last_octave):
    weights=[10,10,80,20,20]
    weights[last_octave - 2] *= 2
    return random.choices([2,3,4,5,6],weights, k=1)[0]

def check_duration(column, start, array, threshold):
    duration = 0
    for future_note in array[start:start+64,column]:
        duration += 1
        if(future_note < 1000):# threshold * 0.8):
            return duration
    return duration
        
def nearest_pow_2(num):
    while(np.log2(num) % 1 != 0):
        num += 1
    return num
    
def average_velocity(column, start, end, array):
    return np.mean(array[start:end,column])
    

def scale(value):
    return (value / 5.0) * MAX_VELOCITY

song_meter = (64,16) # 4:4 in 60pm, 40:40 in 600bpm
i = 0
from matplotlib import pyplot as plt
for data in datas:
    # update values according to scale to represent velocity
    data = scale(data)
    
    on_off = (data > ((1.2/5.0)*MAX_VELOCITY)).astype(int)
    #plt.imshow(on_off)
    #plt.show()
    #break
    data = data * on_off
    
    ttl = 0
    count = 0
    for row in data:
        for val in row:
            if(val != 0):
                count += 1
                ttl += val
    print(ttl/count)
    threshold = ttl/count * .5
    comp = Composition()
    last_oct = 4
    for note in notes_list:
        t = Track(MidiInstrument())
        t.instrument.midi_instr = i
        
        for barnum in range(125 // 64):
            b = Bar()
            b.set_meter(song_meter)
            pos = 0
            while pos < 64:
                last_oct = choose_octave(last_oct)
                dur = nearest_pow_2(check_duration(i, barnum*64+pos, data,threshold))
                to_play = Note(note,last_oct)
                vel = int(data[barnum * 64 + pos,i])
                to_play.set_velocity(vel)
                #dur = check_duration(i, pos, data)
                b.place_notes(to_play,16/dur)
                pos += dur
            #print(b.length)    
            t.add_bar(b)
        '''
        nextPos = 0
        pos = 0
        rest_dur = 0
        while(pos < 255):
            if(data[pos,i] == 0):
                rest_dur += 1
                pos += 1
            else:
                dur = check_duration(i,pos,data)
                if(rest_dur > 0):
                    b.place_rest(16/rest_dur)
                rest_dur = 0
                velocity = average_velocity(i,pos,pos+dur,data)
                last_oct = choose_octave(last_oct)
                to_play = Note(note, last_oct)
                to_play.set_velocity(int(velocity))
                #maybe notecontainer to include other octaves
                b.place_notes(to_play, 16/dur)
                pos += dur
        '''
        filename = "song{0}.mid".format(i)
        i += 1
        #print(b)
        
        comp.add_track(t)
    midi_file_out.write_Composition(filename,comp,bpm = int(600/16))