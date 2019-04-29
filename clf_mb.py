import scipy.io
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
import numpy as np
from helper_funcs import *
import matplotlib.pyplot as plt
from sklearn.ensemble import BaggingClassifier
from sklearn.neural_network import MLPClassifier
import time
import datetime
'''
Final submission functions
'''


n = [1]
Err_test = np.zeros(len(n))
Err_train = np.zeros(len(n))

for patient in range(1,2):
    Xtrn, ytrn = get_train(patient)
    Xval, yval = get_val(patient)
        
    for i in range(0,len(n)):
        #clf = BaggingClassifier(base_estimator = DecisionTreeClassifier(criterion='entropy',max_depth = 5), n_estimators = n[i])
        #clf= DecisionTreeClassifier(criterion='gini',max_depth=depth[i])
        #clf = MLPClassifier(solver='sgd',activation = 'tanh', alpha=1e-4, hidden_layer_sizes=n[i])
        t = time.time()
        clf = SVC(kernel = 'rbf', C = 10, gamma = 10)
        #start w auto
        #2nd run C=1 gamma = 1
        #C=1, gamma = 0.01
        #c = .01, gamma = 0.01
        #c = 0.1, gamma = 10
        #10,10
        # - test error for all combos 0.73170132
        print('start fit')
        clf.fit(Xtrn, ytrn)
        elapsed = (time.time()-t)/60
        print(elapsed)
        
        print('start train predict')
        y_trn_pred = clf.predict(Xtrn)
        elapsed = (time.time()-t)/60
        print(elapsed)
        print('start test predict')
        y_val_pred = clf.predict(Xval)
        elapsed = (time.time()-t)/60
        print(elapsed)
        Err_train[i] = accuracy_score(ytrn,y_trn_pred)
        Err_test[i] = accuracy_score(yval,y_val_pred)
        
    indices = 1#range(0,len(n))
    
    print(Err_train)
    print(Err_test)
    plt.plot(indices,Err_train, label = "training")
    plt.plot(indices,Err_test, label = "testing")
    plt.legend()
        
        
        
        #classifiers.append(clf)        
                

