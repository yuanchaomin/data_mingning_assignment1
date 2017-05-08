import pandas as pd
import numpy as np
import logistic_classifier as lc
import compute_confusion_matrix2 as cc
from numpy import linalg as LA
import os

cwd = os.getcwd()

X_test_address = cwd + '/data/temp/X_raw_test.npy'
y_test_address = cwd + '/data/temp/X_test_label.npy'
V_address = cwd + '/data/temp/all_V_1000.npy'
unique_label_address = cwd + '/data/temp/unique_label'

X_raw_test = np.load(X_test_address)
y_test = np.load(y_test_address)
V_1000 = np.load(V_address)

#V_1000 = np.load('C:/Users/Chaomin/Desktop/data_mining/data/result/V_1000.npy')
label = np.unique(y_test)
#X_test = np.dot(X_raw_test,V_1000.T)

m_test, n_test = np.shape(X_raw_test)
X_raw_test = X_raw_test.astype(float)

y_test = y_test.astype(str)

unique_label = np.unique(y_test)
np.save(unique_label_address, unique_label)

# print('start SVD!')
# U,s,V = LA.svd(X_test_data, full_matrices=False)
#
# print('SVD finish!')
#
# V_1000 = V[:1000,:]
X_test = np.dot(X_raw_test, V_1000.T)

for i in range(5):
    logit_cl  = lc.LogisticClassifier(label)
    weights_address = cwd + '/data/result/weight_data/weight' + str(i) +'.npy'

    test_result_dir = cwd + '/data/result/test'

    if not os.path.exists(test_result_dir):
        os.makedirs(test_result_dir)

    matrix_withheader_address = cwd + '/data/result/test/confusion_matrix_with_header' + str(i) + '.csv'
    matrix_address =  cwd + '/data/result/test/confusion_matrix' + str(i) + '.csv'
    acc_matrix_address = cwd + '/data/result/test/acc' + str(i) + '.csv'


    weights = np.load(weights_address)
    prob, in_matrix = logit_cl.predict_prob(X_test, weights, multiclass=True)
    result_list = logit_cl.predict_label(prob, 0.5, multiclass=True)
    predict_result_array = np.array(result_list, dtype=str)


    confusion_matrix_builder = cc.confusion_matrix_builder(y_test, predict_result_array, label)
    confusion_matrix, label_list = confusion_matrix_builder.compute_confusion_matrix()
    confusion_matrix_builder.save_file(confusion_matrix, label_list, matrix_withheader_address, matrix_address)
    acc_matrix = cc.calculate_r_c_sum(confusion_matrix)

    row_header_l = list(['binary_classifier_by_label', 'TP', 'FN', 'FP', 'TN', 'ACC', 'SPE', 'PRE', 'F1'])
    column_extra_string = 'binary_classifier_by_label'
    result_df = cc.calculate_acc(confusion_matrix, acc_matrix, row_header_l, column_extra_string, label_list, data_sample_size=len(y_test))
    # result_df.to_csv('C:/Users/Chaomin/Desktop/data_mining/data/intermediate/result/acc.csv', index=False)

    result_df.to_csv(acc_matrix_address, index=False)
    print('predict label based on model {0} finishes !'.format(i + 1))

