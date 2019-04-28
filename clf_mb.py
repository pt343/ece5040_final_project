import scipy.io
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
from helper_funcs import *

'''
Final submission functions
'''

if __name__=='__main__':
    
    # load train data
    classifiers=[]
    for patient in range(1,2):
        Xtrn, ytrn = get_train(patient)
        Xval, yval = get_val(patient)
        

        clf= SVC(gamma='auto',kernel = 'rbf')
        clf.fit(Xtrn, ytrn)
        
        y_trn_pred = clf.predict(Xtrn)
        y_val_pred = clf.predict(Xval)
        
        Err_train = accuracy_score(y_trn,y_trn_pred)
        Err_test = accuracy_score(y_val,y_val_pred)
        
        print(Err_test)
        print(Err_train)
        
        
        
        #classifiers.append(clf)        
                

        
'''        
        
for i in range(0,len(indices)):
    bagging = BaggingClassifier(base_estimator = DecisionTreeClassifier(criterion='entropy'), n_estimators = indices[i])
    bagging.fit(X_train,y_train)
    y_train_pred = bagging.predict(X_train)
    Err_Train[i] = accuracy_score(y_train,y_train_pred)
    
    y_test_pred = bagging.predict(X_test)
    Err_Test[i] = accuracy_score(y_test,y_test_pred)
            
'''

