import numpy as np
import matplotlib.pyplot as plt
from helper_funcs import train_tree, predict_test


if __name__=='__main__':
    save_file = ''

    average_train_errors = []
    average_val_errors = []
    average_auc_vals1 = []
    average_auc_vals2 = []
    average_auc_vals3 = []
    average_auc_vals4 = []
    average_auc_vals5 = []
    average_auc_vals6 = []
    average_auc_vals7 = []

    # sweep a variable for optimization
    var_sweep = [i for i in range(1,20)]

    patient = 2

    # optimal max depths for each patient, calculated graphically
    max_depth = [4, 4, 4, 4, None, 4, 4]
    min_samples_leaf = [1, 1, 1, 1, 2, 1, 1]


    # for i in var_sweep:
    #     # train the tree with given parameters
    #     classifiers, train_errors, val_errors, auc_vals = train_tree(min_samples_leaf=i)
    #     #average_train_errors.append(np.mean(train_errors))
    #     #average_val_errors.append(np.mean(val_errors))
    #     average_train_errors.append(train_errors[patient-1])
    #     average_val_errors.append(val_errors[patient-1])
    #     average_auc_vals1.append(auc_vals[0])
    #     average_auc_vals2.append(auc_vals[1])
    #     average_auc_vals3.append(auc_vals[2])
    #     average_auc_vals4.append(auc_vals[3])
    #     average_auc_vals5.append(auc_vals[4])
    #     average_auc_vals6.append(auc_vals[5])
    #     average_auc_vals7.append(auc_vals[6])

    classifiers, train_errors, val_errors, auc_vals = train_tree(presort=True)

    if save_file != '':
        predict_test(save_file, classifiers)

    # print('average train error={}'.format(np.mean(train_errors)))
    # print('average val error={}'.format(np.mean(val_errors)))
    print('auc={}'.format(np.mean(auc_vals)))

    #print('train errors:')
    #print(average_train_errors)
    #print('average train errors = {}'.format(np.mean(average_train_errors)))
    #print('val errors:')
    #print(average_val_errors)
    #print('average val errors = {}'.format(np.mean(average_val_errors)))

    # legend = []
    # for i in min_samples_leaf:
    # #plt.plot(min_samples_leaf, average_train_errors)
    #     plt.plot(min_samples_leaf, average_val_errors[i:i+18])
    #     legend.append('min_samples_leaf=' + str(i))
    # plt.xlabel('max_depth')
    # plt.ylabel('error')
    # plt.legend(legend)
    # plt.show()

    # plt.plot(var_sweep,average_train_errors)
    # plt.plot(var_sweep,average_val_errors)
    # # plt.xlabel('min_samples_leaf')
    # plt.ylabel('error')
    # plt.legend(['train error', 'val error'])
    # plt.show()

    # plt.plot(var_sweep, average_auc_vals1)
    # plt.plot(var_sweep, average_auc_vals2)
    # plt.plot(var_sweep, average_auc_vals3)
    # plt.plot(var_sweep, average_auc_vals4)
    # plt.plot(var_sweep, average_auc_vals5)
    # plt.plot(var_sweep, average_auc_vals6)
    # plt.plot(var_sweep, average_auc_vals7)
    # plt.legend(['pt1', 'pt2', 'pt3', 'pt4', 'pt5', 'pt6', 'pt7'])
    # plt.show()

