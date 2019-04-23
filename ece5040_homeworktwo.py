#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 16:29:44 2019

@author: yijiaosong
"""
import numpy as np
import scipy as sp
import matplotlib.pyplot as plot

data=sp.io.loadmat('/Users/yijiaosong/Downloads/data.mat')

channels=data['data']
"""fig = plot.figure()

plot.subplot(511)
plot.plot(channels[:,13])
plot.title('channel 13')

plot.subplot(512)
plot.plot(channels[:,18])
plot.title('channel 18')

plot.subplot(513)
plot.plot(channels[:,19])
plot.title('channel 19')

plot.subplot(514)
plot.plot(channels[:,20])
plot.title('channel 20')

plot.subplot(515)
plot.plot(channels[:,25])
plot.title('channel 25')
plot.show()
"""

# Line-length
def line_length(chan_num):
    line_length=[]
    for start_index in range(0,len(channels[:,chan_num])-5000,5000):
        total=0
        index=start_index
        while index<start_index+5000:
          total=total+abs(channels[index,chan_num]-channels[index+1,chan_num]) 
          index=index+1
        line_length.append(total)
        
         
    return np.array(line_length)        



# Energy

def get_energy(chan_num):
    print(chan_num)
    squared=[int(i ** 2) for i in channels[:,chan_num]]
   
    energy=[]
    
    for start_index in range(0,len(channels[:,chan_num])-5000,5000):
        total=np.sum(squared[start_index:start_index+5000])
        energy.append(total)
    return energy





# Variance

def get_variance(chan_num):
    
    variance=[]
    
    for start_index in range(0,len(channels[:,chan_num])-5000,5000):
        total=np.var(channels[start_index:start_index+5000,chan_num])
        variance.append(total)
    return variance



# Spectral power in the following band: Beta: 12–30 Hz
# Spectral power in the following band: HFO: 100–600 Hz
def get_power_spec(chan_num):
    ps_beta=[]
    ps_hfo=[]
    for start_index in range(0,len(channels[:,chan_num])-5000,5000):
        fft=np.abs(np.fft.fft(channels[start_index:start_index+5000,chan_num]))
        #freq = np.fft.fftfreq(5000, d=1/5000)
        #plot.plot(freq,fft)
       
        ps_beta.append(np.sum(fft[12:31]))
        ps_hfo.append(np.sum(fft[100:601]))
    return  ps_beta, ps_hfo     





#RUN
channels_plot= [13,18,19,20,25]

for i in channels_plot:
    fig = plot.figure(figsize=(10,25))

   
    plot.subplot(611)
    plot.plot(channels[:,i])
    plot.title('Channel_'+str(i))
    
    ll=line_length(i)
    plot.subplot(612)
    plot.plot(range(0,1372633-5000,5000),ll)
    plot.title('Line Length')
    
    energy=get_energy(i)
    plot.subplot(613)
    plot.plot(range(0,1372633-5000,5000),energy)
    plot.title('Energy')
    
    variance=get_variance(i)
    plot.subplot(614)
    plot.plot(range(0,1372633-5000,5000),variance)
    plot.title('Variance')
    
    beta,hfo=get_power_spec(i)
    plot.subplot(615)
    plot.plot(range(0,1372633-5000,5000),beta)
    plot.title('Beta')
    
    plot.subplot(616)
    plot.plot(range(0,1372633-5000,5000),hfo)
    plot.title('HFO')
    
    plot.savefig('Channel_'+str(i)+'.jpg')
    
    plot.show()