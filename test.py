# -*- coding: utf-8 -*-
"""
Created on Tue Apr 30 23:06:45 2019

@author: mbobb
"""
import numpy as np
from helper_funcs import *

weights = np.zeros(7)
for patient in range(1,8):
        
        val = get_test(patient)
        s = np.shape(val)
        weights[patient-1] = s[0]
        
print(weights)
weights = weights/sum(weights)

print(weights)