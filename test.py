# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 13:51:50 2019

@author: mbobb
"""

from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot
import numpy as np
import scipy
from scipy import signal
import scipy.io
import os
from helper_funcs import channel_line_length


datafolderpath = 'C:/Users/mbobb/Documents/ece5040'

labels = []
LL = []

for i in range(1,8):
    patient = i
    
    #create path to patient's files
    ictalfilepath = datafolderpath + '/data/patient_'+str(patient)+'/ictal train/'
    nonictalfilepath = datafolderpath + '/data/patient_'+str(patient)+'/non-ictal train/'

    #load file names
    ictalFiles=os.listdir(ictalfilepath)
    nonictalFiles = os.listdir(nonictalfilepath)
    
    #load first file
    data = scipy.io.loadmat(ictalfilepath + 'patient_' + str(patient)+'_1')
    data = data['data']
    
    #determine sampling frequency and number of channels
    s = np.shape(data)
    fs = s[0]
    n_channels = s[1]
    
    LLictal = []
    #loop over every ictal file
    for i in range(1,len(ictalFiles)):
        data = scipy.io.loadmat(ictalfilepath + 'patient_' + str(patient)+ '_' + str(i))
        data = data['data']
    
        #calculate line length for every channel
        for j in range(0,n_channels):
            LLictal = np.append(LLictal,channel_line_length(data[:,j],fs,fs,1))
        
    ictal_labels = np.ones(len(LLictal))
    
    LLnonictal = []
    #loop over every nonictal file
    for i in range(1,len(nonictalFiles)):
        data = scipy.io.loadmat(nonictalfilepath + 'patient_' + str(patient) + '_' + str(i))
        data = data['data']
        
        for j in range(0,n_channels):
            LLnonictal = np.append(LLnonictal,channel_line_length(data[:,j],fs,fs,1))

    nonictal_labels = np.zeros(len(LLnonictal))
    
    labels_sub = np.append(ictal_labels,nonictal_labels)
    LL_sub = np.append(LLictal,LLnonictal)
    
    labels = np.append(labels,labels_sub)
    LL = np.append(LL,LL_sub)
