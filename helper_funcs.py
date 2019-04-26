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


def channel_line_length(x, n, fs, w):
    line_length = []
    num_windows = n//fs #floor
    
    for i in range(0,num_windows):
        ll_sum=0
        for j in range(w*fs*i,w*fs*(i+1)-1):
            ll_sum += abs(x[(j+1)]-x[j])
        line_length = np.append(line_length,ll_sum)
    
    return line_length








    

