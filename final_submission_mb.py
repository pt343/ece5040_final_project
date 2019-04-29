import scipy.io
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
import numpy as np
from sklearn import svm
import csv
from helper_funcs import *

'''
Final submission functions
'''

if __name__=='__main__':
    
    # load train data
    classifiers=[]
   
    for patient in range(1,8):
        Xtrn, ytrn = get_train(patient)
        Xval, yval = get_val(patient)
        
        clf = DecisionTreeClassifier(criterion='entropy',max_depth=5)
        clf.fit(Xtrn, ytrn)
        classifiers.append(clf)             

    #HARDCODE
    channels= [96,56,16,88,104,88,96]
    
    
    f = open("submission_test_mb.csv",'w')
    csvwriter = csv.writer(f)
    csvwriter.writerow(["id", "prediction"])
    row_count=1
    
    for patient in range(1,8):
        
        val = get_test(patient)
        
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
    
    
 