import numpy as np
import math
import scipy
import scipy.io
import os
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import roc_auc_score
import csv

#======================================================================#
# Global variables
#======================================================================#
channels = [96, 56, 16, 88, 104, 88, 96] # number of channels for each patient

#======================================================================#
# Data Functions
#======================================================================#
def reshape_data(arr, patient):
    '''
    Transforms the data so that you are looking at average of every 8 channels
    '''
    div_num = 8 # number of channels to divide by
    x, y = np.shape(arr)
    num_data_pts = x//channels[patient-1]
    print('patient {} has {} data'.format(patient, num_data_pts))

    arr2 = np.reshape(arr, (div_num, y*(x//div_num)))

    # calculate the mean for 8 channels
    arr2 = np.mean(arr2, axis=0)
    arr3 = np.reshape(arr2, (num_data_pts, y*(channels[patient-1]//div_num)))
    return arr3


def get_data_array(patient, file_type):
    """
    file_type = 'train', 'test', or 'val'
    Returns an array with all the ictal and nonictal data as well as labels if necessary
    """

    # read in files of interest, both ictal and nonictal
    if file_type=='test':
        readin = scipy.io.loadmat('patient_' + str(patient) + '_test')

        key = list(readin.keys())

        data = np.asarray([readin[key[3]][0],
                          readin[key[4]][0],
                          readin[key[5]][0],
                          readin[key[6]][0][::2],
                          readin[key[6]][0][1::2]])
        data = np.transpose(data)
        data = reshape_data(data, patient)
        return data
    else:
        readin_ict = scipy.io.loadmat('patient_' + str(patient) + '_ict_' + file_type)
        readin_nonict = scipy.io.loadmat('patient_' + str(patient) + '_nonict_' + file_type)
        ict_key = list(readin_ict.keys())
        nonict_key = list(readin_nonict.keys())

        # generate labels corresponding to ictal or nonictal if labels requested
        ict_label = np.ones(int(readin_ict[ict_key[3]].shape[1]/channels[patient-1]))
        nonict_label = np.zeros(int(readin_nonict[nonict_key[3]].shape[1]/channels[patient-1]))
        labels = np.concatenate((ict_label, nonict_label))

        ict = np.asarray([readin_ict[ict_key[3]][0],
                                readin_ict[ict_key[4]][0],
                                readin_ict[ict_key[5]][0],
                                readin_ict[ict_key[6]][0][::2],
                                readin_ict[ict_key[6]][0][1::2]])
        ict = np.transpose(ict)

        nonict = np.asarray([readin_nonict[nonict_key[3]][0],
                                   readin_nonict[nonict_key[4]][0],
                                   readin_nonict[nonict_key[5]][0],
                                   readin_nonict[nonict_key[6]][0][::2],
                                   readin_nonict[nonict_key[6]][0][1::2]])
        nonict = np.transpose(nonict)
        data = np.concatenate((ict, nonict))
        data = reshape_data(data, patient)

        return data, labels


def get_error(G,Y):
    error = 0
    for i in range(len(G)):
        error += 1 if G[i] != Y[i] else 0
    return 1.0*error/len(G)


def get_auc(true_vals, pred_vals):
    return roc_auc_score(true_vals, pred_vals)


#======================================================================#
# Tree Functions
#======================================================================#
def train_tree(**kwargs):
    # keep track of decision tree for each patient
    classifiers = []

    # keep track of errors
    train_errors = []
    val_errors = []
    auc_vals = []

    for patient in range(1,8):
        # get data
        train, labels = get_data_array(patient, 'train')

        # train tree
        clf = DecisionTreeClassifier(
            criterion               =kwargs['criterion'] if 'criterion' in kwargs else 'gini',
            splitter                =kwargs['splitter'] if 'splitter' in kwargs else 'best',
            max_depth               =kwargs['max_depth'][patient-1] if 'max_depth' in kwargs else None,
            min_samples_split       =kwargs['min_samples_split'] if 'min_samples_split' in kwargs else 2,
            min_samples_leaf        =kwargs['min_samples_leaf'][patient-1] if 'min_samples_leaf' in kwargs else 1,
            min_weight_fraction_leaf=kwargs['min_weight_fraction_leaf'] if 'min_weight_fraction_leaf' in kwargs else 0,
            max_features            =kwargs['max_features'] if 'max_features' in kwargs else None,
            random_state            =kwargs['random_state'] if 'random_state' in kwargs else None,
            max_leaf_nodes          =kwargs['max_leaf_nodes'] if 'max_leaf_nodes' in kwargs else None,
            min_impurity_decrease   =kwargs['min_impurity_decrease'] if 'min_impurity_decrease' in kwargs else 0,
            class_weight            =kwargs['class_weight'] if 'class_weight' in kwargs else None,
            presort                 =kwargs['presort'] if 'presort' in kwargs else False
        )
        clf.fit(train, labels)
        classifiers.append(clf)

        # detect training errors from classifier
        train_predict = clf.predict(train)
        train_error = get_error(train_predict, labels)
        train_errors.append(train_error)

        # validation errors
        val, val_labels = get_data_array(patient, 'val')

        #val_labels = np.mean(val_labels.reshape(int(val_labels.size / channels[patient - 1]), channels[patient - 1]), axis=1)
        val_predict = clf.predict(val)
        #val_predict = np.mean(val_predict.reshape(int(val_predict.size / channels[patient - 1]), channels[patient - 1]), axis=1)
        val_error = get_error(val_predict, val_labels)
        val_errors.append(val_error)


        # get roc
        auc_vals.append(get_auc(val_labels, val_predict))

    return classifiers, train_errors, val_errors, auc_vals

def predict_test(file_name, classifiers):
    f = open(file_name, 'w')
    csvwriter = csv.writer(f)
    csvwriter.writerow(["id", "prediction"])

    for patient in range(1, 8):
        val = get_data_array(patient, 'test')
        predictions = classifiers[patient - 1].predict(val)

        # TODO: fix reshaping of channels
        predictions = np.mean(predictions.reshape(int(predictions.size / channels[patient - 1]), channels[patient - 1]),
                              axis=1)

        index = 0
        for prediction in predictions:
            index = index + 1
            csvwriter.writerow(["patient_" + str(patient) + "_" + str(index), prediction])

    f.close()


#======================================================================#
# Feature Functions
#======================================================================#
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


# Spectral power in the following band: Beta 12-30Hz
# Spectral power in the following band: HFO 100-600Hz
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

