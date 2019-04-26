import scipy.io
from sklearn.tree import DecisionTreeClassifier


'''
Final submission functions
'''

if __name__=='__main__':
    # load prerecorded line-length for all patients
    ll_labels_train = scipy.io.loadmat('LL_labels_train.mat')
    ll_labels_train = ll_labels_train['labels']

    ll_vals_train = scipy.io.loadmat('LL_train.mat')
    ll_vals_train = ll_vals_train['LineLengths']

    # load training data and calculate line length
    ll_vals_test = None

    # train decision tree
    clf = DecisionTreeClassifier(criterion='gini', max_depth=5)
    clf.fit(ll_vals_train, ll_labels_train)

    # calculate error for training data
    ll_predict_train = clf.fit(ll_vals_train)

    # predict data for test values
    ll_predict_test = clf.predict(ll_vals_test)

