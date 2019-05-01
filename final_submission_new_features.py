import scipy.io
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn import svm
import csv
from sklearn.ensemble import RandomForestClassifier

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
        nonict_key=list(readin_nonict.keys())
        
        ict_label=np.ones(int(readin_ict[ict_key[3]].shape[1]))
        nonict_label=np.zeros(int(readin_nonict[nonict_key[3]].shape[1]))
        
        
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
        
        
        readin_ictval=scipy.io.loadmat('patient_'+str(patient)+'_ict_val')
        readin_nonictval=scipy.io.loadmat('patient_'+str(patient)+'_nonict_val')
        
        ict_keyval=list(readin_ict.keys())
        print(ict_key)
        nonict_keyval=list(readin_nonict.keys())
        
        ict_labelval=np.ones(int(readin_ictval[ict_key[3]].shape[1]))
        nonict_labelval=np.zeros(int(readin_nonictval[nonict_key[3]].shape[1]))
        labels=np.concatenate((ict_label, nonict_label, ict_labelval,nonict_labelval))
        
        val_ict=np.asarray([readin_ictval[ict_keyval[3]][0], 
               readin_ictval[ict_keyval[4]][0],
               readin_ictval[ict_keyval[5]][0],
               readin_ictval[ict_keyval[6]][0][::2],
               readin_ictval[ict_keyval[6]][0][1::2]])
        val_ict= np.transpose(val_ict)
        
        
        val_nonict=np.asarray([readin_nonictval[nonict_keyval[3]][0], 
               readin_nonictval[nonict_keyval[4]][0],
               readin_nonictval[nonict_keyval[5]][0],
               readin_nonictval[nonict_keyval[6]][0][::2],
               readin_nonictval[nonict_keyval[6]][0][1::2]])
        val_nonict= np.transpose(val_nonict)
        
        
        
        train= np.concatenate((train_ict, train_nonict, val_ict, val_nonict))
        """if patient==1:
            save_training=train
            save_labels=labels
            
        else: 
            save_training=np.concatenate((save_training, train))
            save_labels=np.concatenate((save_labels, labels))"""
        
        clf= RandomForestClassifier()
        clf.fit(train, labels)
        classifiers.append(clf) 
        

    # load val data, and test
    
    """for patient in range(1,8):
        
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
        predictions=classifiers[patient].predict(val)"""
        
    
    #HARDCODE
    """combined= DecisionTreeClassifier(criterion='entropy')
    combined.fit(save_training, save_labels)"""
    
    channels= [96,56,16,88,104,88,96]
    
    
    f = open("submission_test_rf.csv",'w')
    csvwriter = csv.writer(f)
    csvwriter.writerow(["id", "prediction"])
    row_count=1
    
    for patient in range(1,8):
        
        readin=scipy.io.loadmat('patient_'+str(patient)+'_test')
        
        key=list(readin.keys())
        
        val=np.asarray([readin[key[3]][0], 
               readin[key[4]][0],
               readin[key[5]][0],
               readin[key[6]][0][::2],
               readin[key[6]][0][1::2]])
        val= np.transpose(val)
        print(patient)
        predictions=classifiers[patient-1].predict(val)
        #predictions=combined.predict(val)
        predictions=np.mean(predictions.reshape(int(predictions.size/channels[patient-1]), channels[patient-1]), axis=1)
        
       
        index=0
        for prediction in predictions:
            index=index+1
            csvwriter.writerow(["patient_"+str(patient)+"_"+str(index), prediction])
            row_count=row_count+1
        print(row_count)
        #print(index)
    f.close()
    
    
 