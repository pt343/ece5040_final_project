import scipy.io
from sklearn.tree import DecisionTreeClassifier
import numpy as np
from sklearn import svm
import csv
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt

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
        ict_key.sort()
        nonict_key=list(readin_nonict.keys())
        nonict_key.sort()
        
        ict_label=np.ones(int(readin_ict[ict_key[3]].shape[1]))
        nonict_label=np.zeros(int(readin_nonict[nonict_key[3]].shape[1]))
        
        readin_ict2=scipy.io.loadmat('patient_'+str(patient)+'_ict_train2')
        readin_nonict2=scipy.io.loadmat('patient_'+str(patient)+'_nonict_train2')
        
        ict_key2=list(readin_ict2.keys())
        ict_key2.sort()
        nonict_key2=list(readin_nonict2.keys())
        ict_key2.sort()
        
        
        train_ict=np.asarray([readin_ict[ict_key[3]][0], 
               readin_ict[ict_key[4]][0],
               readin_ict[ict_key[6]][0],
               readin_ict[ict_key[5]][0][::2],
               readin_ict[ict_key[5]][0][1::2],
               readin_ict2[ict_key2[3]][0], 
               readin_ict2[ict_key2[4]][0],
               readin_ict2[ict_key2[5]][0]])
    
        train_ict= np.transpose(train_ict)
        
        
        train_nonict=np.asarray([readin_nonict[nonict_key[3]][0], 
               readin_nonict[nonict_key[4]][0],
               readin_nonict[nonict_key[6]][0],
               readin_nonict[nonict_key[5]][0][::2],
               readin_nonict[nonict_key[5]][0][1::2],
               readin_nonict2[nonict_key2[3]][0], 
               readin_nonict2[nonict_key2[4]][0],
               readin_nonict2[nonict_key2[5]][0]])
    
        train_nonict= np.transpose(train_nonict)
        
        
        readin_ictval=scipy.io.loadmat('patient_'+str(patient)+'_ict_val')
        readin_nonictval=scipy.io.loadmat('patient_'+str(patient)+'_nonict_val')
        
        ict_keyval=list(readin_ict.keys())
        ict_keyval.sort()
        nonict_keyval=list(readin_nonict.keys())
        nonict_keyval.sort()
        
        ict_labelval=np.ones(int(readin_ictval[ict_key[3]].shape[1]))
        nonict_labelval=np.zeros(int(readin_nonictval[nonict_key[3]].shape[1]))
        labels=np.concatenate((ict_label, nonict_label, ict_labelval,nonict_labelval))
        
        readin_ictval2=scipy.io.loadmat('patient_'+str(patient)+'_ict_val2')
        readin_nonictval2=scipy.io.loadmat('patient_'+str(patient)+'_nonict_val2')
        ict_keyval2=list(readin_ict2.keys())
        ict_keyval2.sort()
        nonict_keyval2=list(readin_nonict2.keys())
        ict_keyval2.sort()
        
        
        labels=np.concatenate((ict_label,nonict_label, ict_labelval,nonict_labelval))
        
        val_ict=np.asarray([readin_ictval[ict_keyval[3]][0], 
               readin_ictval[ict_keyval[4]][0],
               readin_ictval[ict_keyval[6]][0],
               readin_ictval[ict_keyval[5]][0][::2],
               readin_ictval[ict_keyval[5]][0][1::2],
               readin_ictval2[ict_keyval2[3]][0], 
               readin_ictval2[ict_keyval2[4]][0],
               readin_ictval2[ict_keyval2[5]][0]])
        val_ict= np.transpose(val_ict)
        
        
        val_nonict=np.asarray([readin_nonictval[nonict_keyval[3]][0], 
               readin_nonictval[nonict_keyval[4]][0],
               readin_nonictval[nonict_keyval[6]][0],
               readin_nonictval[nonict_keyval[5]][0][::2],
               readin_nonictval[nonict_keyval[5]][0][1::2],
               readin_nonictval2[nonict_keyval2[3]][0], 
               readin_nonictval2[nonict_keyval2[4]][0],
               readin_nonictval2[nonict_keyval2[5]][0]])
        val_nonict= np.transpose(val_nonict)
        
        
        
        train= np.concatenate((train_ict, train_nonict, val_ict, val_nonict))
        
        # plotting histograms
        
        plt.hist(np.transpose(train)[0], bins=50)  # arguments are passed to np.histogram
        plt.title("line length patient "+ str(patient))
        plt.show()
        
        plt.hist(np.transpose(train)[1], bins=100)  # arguments are passed to np.histogram
        plt.title("energy patient "+ str(patient))
        plt.show()
        
        plt.hist(np.transpose(train)[2], bins=100)  # arguments are passed to np.histogram
        plt.title("variance patient "+ str(patient))
        plt.show()
        
        
        plt.hist(np.transpose(train)[3], bins=100)  # arguments are passed to np.histogram
        plt.title("powerspec beta patient "+ str(patient))
        plt.show() 
        
        plt.hist(np.transpose(train)[4], bins=100)  # arguments are passed to np.histogram
        plt.title("power_spec hfo patient "+ str(patient))
        plt.show()
        
        plt.hist(np.transpose(train)[5], bins=100)  # arguments are passed to np.histogram
        plt.title("kurtosis patient "+ str(patient))
        plt.show()
        
        plt.hist(np.transpose(train)[6], bins=100)  # arguments are passed to np.histogram
        plt.title("mean patient "+ str(patient))
        plt.show()
        
        
        plt.hist(np.transpose(train)[7], bins=100)  # arguments are passed to np.histogram
        plt.title("skew patient "+ str(patient))
        plt.show()
        
    

        
        
        train= np.concatenate((train_ict, train_nonict, val_ict, val_nonict))
        clf= DecisionTreeClassifier(criterion='entropy')
        clf.fit(train, labels)
        classifiers.append(clf) 
        

    #HARDCODE
    """combined= DecisionTreeClassifier(criterion='entropy')
    combined.fit(save_training, save_labels)"""
    
    channels= [96,56,16,88,104,88,96]
    
    
    f = open("submission_test_new_features.csv",'w')
    csvwriter = csv.writer(f)
    csvwriter.writerow(["id", "prediction"])
    row_count=1
    
    for patient in range(1,8):
        
        readin=scipy.io.loadmat('patient_'+str(patient)+'_test')
        
        key=list(readin.keys())
        key.sort()
        
        readin2=scipy.io.loadmat('patient_'+str(patient)+'_test2')
        
        key2=list(readin2.keys())
        key2.sort()
        print(readin[key[3]][0]) 
               
        
        val=np.asarray([readin[key[3]][0], 
               readin[key[4]][0],
               readin[key[6]][0],
               readin[key[5]][0][::2],
               readin[key[5]][0][1::2],
               readin2[key2[3]][0], 
               readin2[key2[4]][0],
               readin2[key2[5]][0]])
    
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
    
    
 