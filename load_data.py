# -*- coding: utf-8 -*-
"""
Created on Tue Apr 23 15:40:31 2019

@author: mbobb
"""
from sklearn.tree import DecisionTreeClassifier
from matplotlib import pyplot
import numpy as np
import scipy
from scipy import signal
import scipy.io
import os

def channel_line_length(x, n, fs, w):
    line_length = []
    num_windows = n//fs #floor
    
    for i in range(0,num_windows):
        ll_sum=0
        for j in range(w*fs*i,w*fs*(i+1)-1):
            ll_sum += abs(x[(j+1)]-x[j])
        line_length = np.append(line_length,ll_sum)
    
    return line_length



datafolderpath = 'C:/Users/mbobb/Documents/ece5040'
patient = 3
ictalfilepath = datafolderpath + '/data/patient_'+str(patient)+'/ictal train/'
nonictalfilepath = datafolderpath + '/data/patient_'+str(patient)+'/non-ictal train/'

ictalFiles=os.listdir(ictalfilepath)
nonictalFiles = os.listdir(nonictalfilepath)

data = scipy.io.loadmat(ictalfilepath + 'patient_' + str(patient)+'_1')
data = data['data']
s = np.shape(data)

fs = s[0]
n_channels = s[1]


LLictal = []

for i in range(0,len(ictalFiles)):
    data = scipy.io.loadmat(ictalfilepath + 'patient_' + str(patient)+ '_1')
    data = data['data']
    
    for j in range(0,n_channels):
        LLictal = np.append(LLictal,channel_line_length(data[:,j],fs,fs,1))
        
ictal_labels = np.ones(len(LLictal))

LLnonictal = []

for i in range(0,len(nonictalFiles)):
    data = scipy.io.loadmat(nonictalfilepath + 'patient_' + str(patient) + '_1')
    data = data['data']
        
    for j in range(0,n_channels):
        LLnonictal = np.append(LLnonictal,channel_line_length(data[:,j],fs,fs,1))

nonictal_labels = np.zeros(len(LLnonictal))

labels = np.append(ictal_labels,nonictal_labels)
features = np.append(LLictal,LLnonictal).reshape(-1,1)

clf = DecisionTreeClassifier(max_depth = 5)
clf.fit(features,labels)
label_pred_train = clf.predict(features)

err_train = sum(label_pred_train.astype(int)^labels.astype(int))/len(labels)




    


'''
data1 = scipy.io.loadmat('C:/Users/mbobb/Documents/ece5040/data/patient_3/ictal train/patient_3_1.mat')
data1 = data1['data']

def channel_line_length(x, n, fs, w):
    line_length = []
    num_windows = n//fs #floor
    
    for i in range(0,num_windows):
        ll_sum=0
        for j in range(w*fs*i,w*fs*(i+1)-1):
            ll_sum += abs(x[(j+1)]-x[j])
        line_length = np.append(line_length,ll_sum)
    
    return line_length

for i in range
'''