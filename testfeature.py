# -*- coding: utf-8 -*-
"""
Created on Wed May  1 21:46:13 2019

@author: mbobb
"""
import scipy.io
from scipy.stats import entropy
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import hilbert, chirp


data = scipy.io.loadmat('C:/Users/mbobb/Documents/ece5040/data/patient_1/ictal train/patient_1_1.mat')
data = data['data']

#determine sampling frequency and number of channels
s = np.shape(data)
fs = s[0]
n_channels = s[1]

signal = data[:,0]

analytic_signal = hilbert(signal)
instantaneous_phase = np.unwrap(np.angle(analytic_signal))
instantaneous_frequency = (np.diff(instantaneous_phase)/(2.0*np.pi)*fs)

ent = entropy(signal)


