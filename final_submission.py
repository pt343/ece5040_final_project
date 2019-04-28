import scipy.io
from sklearn.tree import DecisionTreeClassifier
import numpy as np

'''
Final submission functions
'''

if __name__=='__main__':
    
    # load train data
    classifiers=[]
    for patient in range(1,8):
        readin_ict=scipy.io.loadmat('patient_'+str(patient)+'_ict_train')
        readin_nonict=scipy.io.loadmat('patient_'+str(patient)+'_nonict_train')
        
        ict_key=list(readin_ict.keys())
        print(ict_key)
        nonict_key=list(readin_nonict.keys())
        
        ict_label=np.ones(int(readin_ict[ict_key[3]].shape[1]))
        nonict_label=np.zeros(int(readin_nonict[nonict_key[3]].shape[1]))
        labels=np.concatenate((ict_label,nonict_label))
        
        train_ict=np.asarray([readin_ict[ict_key[3]][0], 
               readin_ict[ict_key[4]][0],
               readin_ict[ict_key[5]][0],
               readin_ict[ict_key[6]][0][::2],
               readin_ict[ict_key[6]][0][1::2]])
        train_ict= np.transpose(train_ict)
        
        
        train_nonict=np.asarray([readin_nonict[nonict_key[3]][0], 
               readin_nonict[nonict_key[4]][0],
               readin_nonict[nonict_key[5]][0],
               readin_nonict[nonict_key[6]][0][::2],
               readin_nonict[nonict_key[6]][0][1::2]])
        train_nonict= np.transpose(train_nonict)
        
        train= np.concatenate((train_ict, train_nonict))
        
        clf= DecisionTreeClassifier(criterion='gini')
        clf.fit(train, labels)
        classifiers.append(clf)        
        
        
    # load val data, and test
    
    for patient in range(1,3):
        
        readin_ict=scipy.io.loadmat('patient_'+str(patient)+'_ict_val')
        readin_nonict=scipy.io.loadmat('patient_'+str(patient)+'_nonict_val')
        
        ict_key=list(readin_ict.keys())
        print(ict_key)
        nonict_key=list(readin_nonict.keys())
        
        ict_label=np.ones(int(readin_ict[ict_key[3]].shape[1]))
        nonict_label=np.zeros(int(readin_nonict[nonict_key[3]].shape[1]))
        labels=np.concatenate((ict_label,nonict_label))
        
        val_ict=np.asarray([readin_ict[ict_key[3]][0], 
               readin_ict[ict_key[4]][0],
               readin_ict[ict_key[5]][0],
               readin_ict[ict_key[6]][0][::2],
               readin_ict[ict_key[6]][0][1::2]])
        val_ict= np.transpose(val_ict)
        
        
        val_nonict=np.asarray([readin_nonict[nonict_key[3]][0], 
               readin_nonict[nonict_key[4]][0],
               readin_nonict[nonict_key[5]][0],
               readin_nonict[nonict_key[6]][0][::2],
               readin_nonict[nonict_key[6]][0][1::2]])
        val_nonict= np.transpose(val_nonict)
        
        val= np.concatenate((val_ict, val_nonict))
        predictions=[]
        for clf in classifiers:
           predictions.append(clf.predict(val))
            


