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
    # var_sweep = [i*5 for i in range(1,8)]

    # optimal max depths for each patient, calculated graphically
    # max_depth = [4, 4, 4, 4, None, 4, 4]
    # min_samples_leaf = [1, 1, 1, 1, 2, 1, 1]

    #
    # for i in var_sweep:
    #     print('sweep = {}'.format(i))
    #     # train the tree with given parameters
    #     classifiers, train_errors, val_errors, auc_vals = train_tree(max_depth=i)
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

    classifiers, train_errors, val_errors, auc_vals = train_tree()

    if save_file != '':
        predict_test(save_file, classifiers)


    # plt.plot(var_sweep,average_train_errors)
    # plt.plot(var_sweep,average_val_errors)
    # # plt.xlabel('min_samples_leaf')
    # plt.ylabel('error')
    # plt.legend(['train error', 'val error'])
    # plt.show()
    #
    # plt.plot(var_sweep, average_auc_vals1)
    # plt.plot(var_sweep, average_auc_vals2)
    # plt.plot(var_sweep, average_auc_vals3)
    # plt.plot(var_sweep, average_auc_vals4)
    # plt.plot(var_sweep, average_auc_vals5)
    # plt.plot(var_sweep, average_auc_vals6)
    # plt.plot(var_sweep, average_auc_vals7)
    # plt.legend(['pt1', 'pt2', 'pt3', 'pt4', 'pt5', 'pt6', 'pt7'])
    # plt.xlabel('Max Depth')
    # plt.ylabel('AUC')
    # plt.show()


    # ROC curves already plotted in tree function,
    # so just need to add labels add line and legend and show plot
    plt.plot([0, 1], [0, 1], '--k')
    plt.legend([
        'Patient 1, AUC={:.3f}'.format(auc_vals[0]),
        'Patient 2, AUC={:.3f}'.format(auc_vals[1]),
        'Patient 3, AUC={:.3f}'.format(auc_vals[2]),
        'Patient 4, AUC={:.3f}'.format(auc_vals[3]),
        'Patient 5, AUC={:.3f}'.format(auc_vals[4]),
        'Patient 6, AUC={:.3f}'.format(auc_vals[5]),
        'Patient 7, AUC={:.3f}'.format(auc_vals[6])
    ])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC for Each Patient')
    plt.show()


