import scipy.io
from sklearn.tree import DecisionTreeClassifier
import numpy as np
import csv



if __name__=='__main__':
    
    # load train data
    classifiers=[]

    # keep track of errors
    train_errors = []
    val_errors = []
   
    for patient in range(1,8):
        readin_ict=scipy.io.loadmat('patient_'+str(patient)+'_ict_train')
        readin_nonict=scipy.io.loadmat('patient_'+str(patient)+'_nonict_train')
        
        ict_key=list(readin_ict.keys())
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



        # detect errors from classifier

        # training errors
        train_predict = clf.predict(train)
        train_error = clf.get_error(train_predict,labels)
        train_errors.append(train_error)

        # held-out errors
        readin_val_ict=scipy.io.loadmat('patient_'+str(patient)+'_ict_val')
        readin_val_nonict = scipy.io.loadmat('patient_' + str(patient) + '_nonict_val')
        val_ict_key=list(readin_val_ict.keys())
        val_nonict_key=list(readin_val_nonict.keys())
        val_ict = np.asarray([readin_val_ict[ict_key[3]][0],
                                readin_ict[ict_key[4]][0],
                                readin_ict[ict_key[5]][0],
                                readin_ict[ict_key[6]][0][::2],
                                readin_ict[ict_key[6]][0][1::2]])
        val_ict = np.transpose(val_ict)
        val_nonict = np.asarray([readin_val_nonict[ict_key[3]][0],
                                readin_ict[ict_key[4]][0],
                                readin_ict[ict_key[5]][0],
                                readin_ict[ict_key[6]][0][::2],
                                readin_ict[ict_key[6]][0][1::2]])
        val_nonict = np.transpose(val_nonict)
        val = np.concatenate(val_ict, val_nonict)
        val_labels = np.concatenate(
            np.ones(int(readin_val_ict[ict_key[3]].shape[1])),
            np.zeros(int(readin_val_nonict[nonict_key[3]].shape[1]))
        )

        val_predict = clf.predict(val)
        val_error = clf.get_error(val_predict, val_labels)
        val_errors.append(val_error)



        

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
    channels= [96,56,16,88,104,88,96]
    
    
    f = open("submission_test.csv",'w')
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
        predictions=np.mean(predictions.reshape(int(predictions.size/channels[patient-1]), channels[patient-1]), axis=1)
        
       
        index=0
        for prediction in predictions:
            index=index+1
            csvwriter.writerow(["patient_"+str(patient)+"_"+str(index), prediction])
            row_count=row_count+1
        print(row_count)
        #print(index)
    f.close()
    
    
    print(train_errors)
    print(val_errors)