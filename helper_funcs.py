# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 13:51:04 2019

@author: mbobb
"""

# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 15:40:31 2019

@author: mbobb
"""

import numpy as np


def channel_line_length(signal):
    index=0
    while index<start_index+5000:
         line_length=line_length+abs(signal[index]-signal[index+1]) 
         index=index+1
    
    return line_length


# Energy

def get_energy(signal):
    squared=[int(i ** 2) for i in signal]
    return np.sum(squared)



# Variance

def get_variance(signal):
    variance=np.var(signal)
    return variance



# Spectral power in the following band: Beta: 12–30 Hz
# Spectral power in the following band: HFO: 100–600 Hz
def get_power_spec(signal):
    fft=np.abs(np.fft.fft(signal))
    ps_beta=np.sum(fft[12:31])
    ps_hfo=np.sum(fft[100:601])
    return  ps_beta, ps_hfo     





    

