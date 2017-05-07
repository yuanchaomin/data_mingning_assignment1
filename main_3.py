import pandas as pd
import numpy as np
import preprocessing
from numpy import linalg as LA
import logistic_classifier as lc
import compute_confusion_matrix2 as cc
import time
import gc

one = time.clock()
start_time = time.clock()
b = []
for i in range(1,13627):
    b.append('feature:'+ str(i))
b.insert(0,'app_name')


print('loading data begins !')
df_appdata = pd.read_csv('C:/Users/Chaomin/Desktop/data_mining/assignment1_2017S1/training_data.csv',names = b)
df_label =pd.read_csv('C:/Users/Chaomin/Desktop/data_mining/assignment1_2017S1/training_labels.csv', names = ['app_name','app_label'])
df_appdata = pd.merge(df_appdata, df_label, on = ['app_name'])
#now, np.shape(result) = (100, 13628)
appdata_matrix = df_appdata.as_matrix()
two = time.clock()
print('time of data_loading = %s \n'%(two - one ))

#np.shape(X_train_ay) == (80,13628) #np.shape(X_test_ay) == (20, 13628)
# The first column of X_train_ay and X_test_ay is the app_name, and the last column is the app_label

print('loading data finish!\n')
print('Spliting data to test_data and training data:')
X_train_ay, X_test_ay, X_train_index, X_test_index = preprocessing.split(appdata_matrix, 0.2, 1)
#np.shape(X_train_ay) == (80,13628) #np.shape(X_test_ay) == (20, 13628)
# The first column of X_train_ay and X_test_ay is the app_name, and the last column is the app_label
m_test, n_test = np.shape(X_test_ay)
m_train, n_train = np.shape(X_train_ay)
X_test_label = X_test_ay[:, n_test - 1]
X_train_label = X_train_ay[:,n_train - 1]
X_test_appname = X_test_ay[:,0]
X_train_appname = X_train_ay[:,0]
X_raw_test = X_test_ay[:, 1:n_test - 1]
X_raw_train = X_train_ay[:, 1: n_train - 1]

#print(np.shape(X_raw_test))   (80,13626)
#print(np.shape(X_raw_train))  (20,13626)

X_raw_test = X_raw_test.astype(float)
X_raw_train = X_raw_train.astype(float)

print('train data SVD starts !\n')
U,s,V = LA.svd(X_raw_train, full_matrices=False)

#s_1000 = s[:1000]
#U_1000 = U[,:1000]
V_1000 = V[:1000,:]

del s
del U
del V
gc.collect()

print('train data SVD finishes !\n')

print('time of data_loading = %s'%(two - one))

print('test data SVD start !\n')
U_t,s_t,V_t = LA.svd(X_raw_test, full_matrices=False)

V_t_1000 = V_t[:1000,:]

del U_t,
del s_t,
del V_t
gc.collect()

print('test data SVD finishes !\n')
three = time.clock()
print('time of SVD = %s'%(three - two))


X_train = np.dot(X_raw_train, V_1000.T)
X_test = np.dot(X_raw_test, V_t_1000.T)
y_train = X_train_label.astype(str)
y_test = X_test_label.astype(str)


for i in range(10):
    print('start {0}-th loop'.format(i + 1))
    three = time.clock()

    label = np.unique(X_train_label)
    label = label.astype(str)

    number_of_fold = 10
    fold_index_list = preprocessing.k_fold_cv_withclass(number_of_fold, X_train_label)
    # for i in range(number_of_fold):
    #         test_index, train_index = preprocessing.arrange_data_in_each_fold(fold_index_list,i,number_of_fold)
    #         X_train = X_raw_train[train_index]
    #         X_test = X_raw_test[test_index]
    #         y_train = X_train_label[train_index]
    #         y_test = X_test_label[test_index]
    #
    #         logit_cl = lc.LogisticClassifier()
    #         weights = logit_cl.fit(X_train, y_train, multiclass=True)
    #         prob, in_matrix = logit_cl.predict_prob(X_test, weights, multiclass=True)
    #         result_list = logit_cl.predict_label(prob, 0.5, multiclass=True)
    #         predict_result_array = np.array(result_list, dtype=str)
    #
    #         matrix_withheader_address = 'C:/Users/Chaomin/Desktop/data_mining/data/intermediate/result/win/confusion_matrix_withheader' + str(i) + '.csv'
    #         matrix_address = 'C:/Users/Chaomin/Desktop/data_mining/data/intermediate/result/win/confusion_matrix' + str(i) + '.csv'
    #         acc_matrix_address = 'C:/Users/Chaomin/Desktop/data_mining/data/intermediate/result/win/acc' + str(i) + '.csv'
    #         confusion_matrix_builder = cc.confusion_matrix_builder(y_test, predict_result_array, label)
    #         confusion_matrix, label_list = confusion_matrix_builder.compute_confusion_matrix()
    #         confusion_matrix_builder.save_file(confusion_matrix, label_list, matrix_withheader_address, matrix_address)
    #         acc_matrix = cc.calculate_r_c_sum(confusion_matrix)
    #
    #         row_header_l = list(['binary_classifier_by_label', 'TP', 'FN', 'FP', 'TN', 'ACC', 'SPE', 'PRE', 'F1'])
    #         column_extra_string = 'binary_classifier_by_label'
    #         result_df = cc.calculate_acc(confusion_matrix, acc_matrix, row_header_l, column_extra_string, label_list,
    #                                      data_sample_size=m_test)
    #         # result_df.to_csv('C:/Users/Chaomin/Desktop/data_mining/data/intermediate/result/acc.csv', index=False)
    #
    #         result_df.to_csv(acc_matrix_address, index=False)
    #
    #         print('Please check that: {0} round'.format(i+1))
    test_index, train_index = preprocessing.arrange_data_in_each_fold(fold_index_list,i, number_of_fold)

    four = time.clock()

    print('{0}-th spliting data finishes!'.format(i +1 ))
    print('time in split fold %s' % (four - three))
    X_in_train = X_train[train_index]
    X_in_test = X_train[test_index]
    y_in_train = y_train[train_index]
    y_in_test = y_train[test_index]

    m_in_test, n_in_test = np.shape(X_in_test)

    print('start training model {0}'.format(i+1))
    matrix_withheader_address = 'C:/Users/Chaomin/Desktop/data_mining/data/intermediate/result/win/confusion_matrix_withheader' + str(i) + '.csv'
    matrix_address = 'C:/Users/Chaomin/Desktop/data_mining/data/intermediate/result/win/confusion_matrix' + str(i) + '.csv'
    acc_matrix_address = 'C:/Users/Chaomin/Desktop/data_mining/data/intermediate/result/win/acc' + str(i) + '.csv'
    weights_address = 'C:/Users/Chaomin/Desktop/data_mining/data/intermediate/result/win/weight' + str(i)
    five = time.clock()

    logit_cl = lc.LogisticClassifier()
    weights = logit_cl.fit(X_in_train, y_in_train, multiclass=True)
    np.save(weights_address,weights)
    prob, in_matrix = logit_cl.predict_prob(X_in_test, weights, multiclass=True)
    result_list = logit_cl.predict_label(prob, 0.5, multiclass=True)
    predict_result_array = np.array(result_list, dtype=str)


    confusion_matrix_builder = cc.confusion_matrix_builder(y_in_test, predict_result_array, label)
    confusion_matrix, label_list = confusion_matrix_builder.compute_confusion_matrix()
    confusion_matrix_builder.save_file(confusion_matrix, label_list, matrix_withheader_address, matrix_address)
    acc_matrix = cc.calculate_r_c_sum(confusion_matrix)

    row_header_l = list(['binary_classifier_by_label', 'TP', 'FN', 'FP', 'TN', 'ACC', 'SPE', 'PRE', 'F1'])
    column_extra_string = 'binary_classifier_by_label'
    result_df = cc.calculate_acc(confusion_matrix, acc_matrix, row_header_l, column_extra_string, label_list, data_sample_size=m_in_test)
    # result_df.to_csv('C:/Users/Chaomin/Desktop/data_mining/data/intermediate/result/acc.csv', index=False)

    result_df.to_csv(acc_matrix_address, index=False)
    six = time.clock()
    print('training model {0} finishes !'.format(i+1))

    del X_in_train
    del X_in_test
    del logit_cl
    del weights
    del prob
    del in_matrix
    del result_list
    del predict_result_array
    del confusion_matrix_builder
    del confusion_matrix
    del label_list

    gc.collect()

    print('Please check that: this is the {0}th round'.format(0 + 1))
    print('time in a loop %s seconds'%(six-five))
