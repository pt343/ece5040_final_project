import scipy.io
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
import numpy as np
from sklearn import svm
import csv
from helper_funcs import *
from sklearn.metrics import accuracy_score
from sklearn.neural_network import MLPClassifier
'''
Final submission functions
'''


if __name__=='__main__':
    
    # load train data
    classifiers=[]


    i = 0
    etrnsum = 0
    etstsum = 0
    weights = [0.0621561,  0.0483696,  0.02670509, 0.13365899, 0.1752466,  0.25189578,
               0.30196783]
    for patient in range(1,8):
       
        Xtrn, ytrn = get_train(patient)
        Xval, yval = get_val(patient)
        
        n = [2,3,4,5,6,7]
        Err_test = np.zeros(len(n))
        Err_train = np.zeros(len(n))
        
        #clf = DecisionTreeClassifier(criterion='entropy')
        #clf = BaggingClassifier(base_estimator = DecisionTreeClassifier(criterion='entropy'), n_estimators = 10)
        clf = MLPClassifier(solver='sgd',activation = 'relu', alpha=1e-4, hidden_layer_sizes=(100))
        clf.fit(Xtrn, ytrn)
        classifiers.append(clf)  
        '''
        y_trn_pred = clf.predict(Xtrn)
        y_val_pred = clf.predict(Xval)
        i+=1
        Err_train = accuracy_score(ytrn,y_trn_pred)
        Err_test = accuracy_score(yval,y_val_pred)
        etrnsum+=Err_train*(weights[patient-1])
        etstsum+=Err_test*(weights[patient-1])
        '''
        print(patient)
        
        #print(Err_train)
        #print(Err_test)
        
    print(etrnsum)
    print(etstsum)
     
    #HARDCODE
    channels= [96,56,16,88,104,88,96]

    f = open("submission_test_mb.csv",'w')
    csvwriter = csv.writer(f)
    csvwriter.writerow(["id", "prediction"])
    row_count=1
    
    for patient in range(1,8):
        
        val = get_test(patient)
        
        #print(patient)
        predictions=classifiers[patient-1].predict(val)
        predictions=np.mean(predictions.reshape(int(predictions.size/channels[patient-1]), channels[patient-1]), axis=1)
        
       
        index=0
        for prediction in predictions:
            index=index+1
            csvwriter.writerow(["patient_"+str(patient)+"_"+str(index), prediction])
            row_count=row_count+1
        #print(row_count)
        #print(index)
    f.close()

 