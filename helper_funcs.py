# -*- coding: utf-8 -*-
"""
<<<<<<< HEAD
Created on Fri Apr 26 13:51:04 2019
@author: mbobb
"""

# -*- coding: utf-8 -*-
"""
=======
>>>>>>> bef985db4b951a4655be801c6e70a137879cbe99
Created on Tue Apr 23 15:40:31 2019
@author: mbobb
"""

import numpy as np
import math
import scipy
from scipy import signal
import scipy.io
import os
import scipy.stats as stats
from scipy.stats import entropy
from scipy.signal import hilbert


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

def get_skew(signal, fs):
    return stats.skew(signal)
    
def get_mean(signal,fs):
    return np.mean(signal)
    
def get_kurtosis(signal, fs): 
    return stats.kurtosis(signal)

def get_entropy(signal,fs):
    return entropy(signal)

def get_IF_mean_var(signal,fs):
    analytic_signal = hilbert(signal)
    instantaneous_phase = np.unwrap(np.angle(analytic_signal))
    instantaneous_frequency = (np.diff(instantaneous_phase)/(2.0*np.pi)*fs)
    mean = np.mean(instantaneous_frequency)
    var = np.var(instantaneous_frequency)
    return mean, var



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
    
    #loop over every feature
    for arg in argv:
        ictaltrainfeatures = []
        
        #loop over every ictal training file
        for i in range(1,num_train_ictal):
            data = scipy.io.loadmat(ictalfilepath + 'patient_' + str(patient)+ '_' + str(i))
            data = data['data']
            data = np.nan_to_num(data)
    
            #calculate each feature for each channel
            for j in range(0,n_channels):
                ictaltrainfeatures = np.append(ictaltrainfeatures,arg(data[:,j],fs))
                
        dict_ictaltrain["feature{0}".format(arg)]=ictaltrainfeatures  
    
    for arg in argv:
        ictalvalfeatures = []
        
        #loop over every ictal validation file        
        for i in range(num_train_ictal,len(ictalFiles)):
            data = scipy.io.loadmat(ictalfilepath + 'patient_' + str(patient)+ '_' + str(i))
            data = data['data']
            data = np.nan_to_num(data)
    
            #calculate each feature for each channel
            for j in range(0,n_channels):
                ictalvalfeatures = np.append(ictalvalfeatures,arg(data[:,j],fs))
                
        dict_ictalval["feature{0}".format(arg)]=ictalvalfeatures 
            
    
    dict_nonictaltrain = {}
    dict_nonictalval = {}
    
    for arg in argv:
        nonictaltrainfeatures = []
        
        #loop over every nonictal train file
        for i in range(1,num_train_nonictal):
            data = scipy.io.loadmat(nonictalfilepath + 'patient_' + str(patient) + '_' + str(i))
            data = data['data']
            data = np.nan_to_num(data)

            for j in range(0,n_channels):
                nonictaltrainfeatures = np.append(nonictaltrainfeatures,arg(data[:,j],fs))
                
        dict_nonictaltrain["feature{0}".format(arg)] = nonictaltrainfeatures
     
        
    for arg in argv:
        nonictalvalfeatures = []
        
        #loop over every nonictal validation file
        for i in range(num_train_nonictal,len(nonictalFiles)):
            data = scipy.io.loadmat(nonictalfilepath + 'patient_' + str(patient) + '_' + str(i))
            data = data['data']
            data = np.nan_to_num(data)
        
            for j in range(0,n_channels):
                nonictalvalfeatures = np.append(nonictalvalfeatures,arg(data[:,j],fs))
                
        dict_nonictalval["feature{0}".format(arg)] = nonictalvalfeatures
        
    
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
    #loop over every feature
    for arg in argv:
        testfeatures = []
        #loop over every ictal training file
        
        for i in range(1,len(files)):
            data = scipy.io.loadmat(filepath + 'patient_' + str(patient)+ '_test_' + str(i))
            print(filepath + 'patient_' + str(patient)+ '_test_' + str(i))
            data = data['data']
            data = np.nan_to_num(data)
    
            #loop over every channel
            for j in range(0,n_channels):
                testfeatures = np.append(testfeatures,arg(data[:,j],fs))
        dict_test["feature{0}".format(arg)]=testfeatures  
    
    return dict_test

def get_train(patient):
        readin_ict=scipy.io.loadmat('patient_'+str(patient)+'_ict_train')
        readin_nonict=scipy.io.loadmat('patient_'+str(patient)+'_nonict_train')
        
        ict_key=list(readin_ict.keys())
        ict_key.sort()
        nonict_key=list(readin_nonict.keys())
        nonict_key.sort()
        
        ict_label=np.ones(int(readin_ict[ict_key[6]].shape[1]))
        nonict_label=np.zeros(int(readin_nonict[nonict_key[6]].shape[1]))
        labels=np.concatenate((ict_label,nonict_label))
        
        train_ict=np.asarray([readin_ict[ict_key[6]][0], 
               readin_ict[ict_key[4]][0],
               readin_ict[ict_key[3]][0],
               readin_ict[ict_key[5]][0][::2],
               readin_ict[ict_key[5]][0][1::2]])
        train_ict= np.transpose(train_ict)
        
        
        train_nonict=np.asarray([readin_nonict[nonict_key[6]][0], 
               readin_nonict[nonict_key[4]][0],
               readin_nonict[nonict_key[3]][0],
               readin_nonict[nonict_key[5]][0][::2],
               readin_nonict[nonict_key[5]][0][1::2]])
        train_nonict= np.transpose(train_nonict)
        
        train= np.concatenate((train_ict, train_nonict))
        
        return train, labels
    
def get_val(patient):
    readin_ict=scipy.io.loadmat('patient_'+str(patient)+'_ict_val')
    readin_nonict=scipy.io.loadmat('patient_'+str(patient)+'_nonict_val')
        
    ict_key=list(readin_ict.keys())
    ict_key.sort()
    nonict_key=list(readin_nonict.keys())
    nonict_key.sort()
        
    ict_label=np.ones(int(readin_ict[ict_key[6]].shape[1]))
    nonict_label=np.zeros(int(readin_nonict[nonict_key[6]].shape[1]))
    labels=np.concatenate((ict_label,nonict_label))
        
    val_ict=np.asarray([readin_ict[ict_key[3]][0], 
               readin_ict[ict_key[4]][0],
               readin_ict[ict_key[6]][0],
               readin_ict[ict_key[5]][0][::2],
               readin_ict[ict_key[5]][0][1::2]])
    val_ict= np.transpose(val_ict)
        
        
    val_nonict=np.asarray([readin_nonict[nonict_key[3]][0], 
               readin_nonict[nonict_key[4]][0],
               readin_nonict[nonict_key[6]][0],
               readin_nonict[nonict_key[5]][0][::2],
               readin_nonict[nonict_key[5]][0][1::2]])
    val_nonict= np.transpose(val_nonict)
        
    val= np.concatenate((val_ict, val_nonict))
    
    return val, labels

def get_test(patient):
    
    readin=scipy.io.loadmat('patient_'+str(patient)+'_test')
        
    key=list(readin.keys())
    key.sort()
        
    val=np.asarray([readin[key[3]][0], 
               readin[key[4]][0],
               readin[key[6]][0],
               readin[key[5]][0][::2],
               readin[key[5]][0][1::2]])
    val= np.transpose(val)
    
    return val
