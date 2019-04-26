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
import math
import scipy
from scipy import signal
import scipy.io
import os


def channel_line_length(signal,fs):
    index=0
    start_index = 0
    line_length = 0
    while index<start_index+(fs-1):
         line_length=line_length+abs(signal[index]-signal[index+1]) 
         index=index+1
    
    return line_length


# Energy

def get_energy(signal,fs):
    squared=[int(i ** 2) for i in signal]
    return np.sum(squared)



# Variance

def get_variance(signal,fs):
    variance=np.var(signal)
    return variance



# Spectral power in the following band: Beta: 12–30 Hz
# Spectral power in the following band: HFO: 100–600 Hz
def get_power_spec(signal,fs):
    fft=np.abs(np.fft.fft(signal))
    ps_beta=np.sum(fft[12:31])
    ps_hfo=np.sum(fft[100:601])
    return  ps_beta, ps_hfo     



def extract_features_train(datafolderpath, patient, *argv):
    
    #create path to patient's files
    ictalfilepath = datafolderpath + '/data/patient_'+str(patient)+'/ictal train/'
    nonictalfilepath = datafolderpath + '/data/patient_'+str(patient)+'/non-ictal train/'
        
    #load file names
    ictalFiles=os.listdir(ictalfilepath)
    nonictalFiles = os.listdir(nonictalfilepath)
    num_train_ictal = math.floor(len(ictalFiles)*0.8)
    num_train_nonictal = math.floor(len(nonictalFiles)*0.8)
    
    #load first file
    data = scipy.io.loadmat(ictalfilepath + 'patient_' + str(patient)+'_1')
    data = data['data']
    
    #determine sampling frequency and number of channels
    s = np.shape(data)
    fs = s[0]
    n_channels = s[1]
    
    dict_ictaltrain = {}
    dict_ictalval = {}
    #loop over every ictal training file
    for i in range(1,num_train_ictal):
        data = scipy.io.loadmat(ictalfilepath + 'patient_' + str(patient)+ '_' + str(i))
        data = data['data']
    
        #calculate each feature for each channel
        count = 0
        for arg in argv:
            count+=1
            ictaltrainfeatures = []
            for j in range(0,n_channels):
                ictaltrainfeatures = np.append(ictaltrainfeatures,arg(data[:,j],fs))
            dict_ictaltrain["feature{0}".format(count)]=ictaltrainfeatures  
    
    #loop over every ictal validation file        
    for i in range(num_train_ictal,len(ictalFiles)):
        data = scipy.io.loadmat(ictalfilepath + 'patient_' + str(patient)+ '_' + str(i))
        data = data['data']
    
        #calculate each feature for each channel
        count = 0
        for arg in argv:
            count+=1
            ictalvalfeatures = []
            for j in range(0,n_channels):
                ictalvalfeatures = np.append(ictalvalfeatures,arg(data[:,j],fs))
            dict_ictalval["feature{0}".format(count)]=ictalvalfeatures 
            
    
    dict_nonictaltrain = {}
    dict_nonictalval = {}
    #loop over every nonictal train file
    for i in range(1,num_train_nonictal):
        data = scipy.io.loadmat(nonictalfilepath + 'patient_' + str(patient) + '_' + str(i))
        data = data['data']
        
        count = 0
        for arg in argv:
            count+=1
            nonictaltrainfeatures = []
        for j in range(0,n_channels):
            nonictaltrainfeatures = np.append(nonictaltrainfeatures,arg(data[:,j],fs))
        dict_nonictaltrain["feature{0}".format(count)] = nonictaltrainfeatures
        
    #loop over every nonictal validation file
    for i in range(num_train_nonictal,len(nonictalFiles)):
        data = scipy.io.loadmat(nonictalfilepath + 'patient_' + str(patient) + '_' + str(i))
        data = data['data']
        
        count = 0
        for arg in argv:
            count+=1
            nonictalvalfeatures = []
        for j in range(0,n_channels):
            nonictalvalfeatures = np.append(nonictalvalfeatures,arg(data[:,j],fs))
        dict_nonictalval["feature{0}".format(count)] = nonictalvalfeatures
        
    
    return dict_ictaltrain, dict_ictalval, dict_nonictaltrain, dict_nonictalval



def extract_features_test(datafolderpath, patient, *argv):
    
    #create path to patient's files
    filepath = datafolderpath + '/data/patient_'+str(patient)+'/test/'
    
    #load file names
    files=os.listdir(filepath)
    
    #load first file
    data = scipy.io.loadmat(filepath + 'patient_' + str(patient)+'_test_1')
    data = data['data']
    
    #determine sampling frequency and number of channels
    s = np.shape(data)
    fs = s[0]
    n_channels = s[1]
    
    dict_test = {}
    #loop over every ictal training file
    for i in range(1,len(files)-1):
        data = scipy.io.loadmat(filepath + 'patient_' + str(patient)+ '_test_' + str(i))
        data = data['data']
    
        #calculate each feature for each channel
        count = 0
        for arg in argv:
            count+=1
            testfeatures = []
            for j in range(0,n_channels):
                testfeatures = np.append(testfeatures,arg(data[:,j],fs))
            dict_test["feature{0}".format(count)]=testfeatures  
    
    return dict_test

