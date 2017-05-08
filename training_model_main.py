import pandas as pd
import numpy as np
import preprocessing
from numpy import linalg as LA
import logistic_classifier as lc
import compute_confusion_matrix2 as cc
import time
import gc
import os


one = time.clock()
start_time = time.clock()

cwd = os.getcwd()

temp_file_directory = cwd + '/data/temp'
if not os.path.exists(temp_file_directory):
    os.makedirs(temp_file_directory)

result_directory = cwd + '/data/result'
if not os.path.exists(temp_file_directory):
    os.makedirs(temp_file_directory)


b = []
for i in range(1,13627):
    b.append('feature:'+ str(i))
b.insert(0,'app_name')


print('loading data begins !')
training_data_address = cwd + '/data/training_data.csv'
training_label_address = cwd + '/data/training_labels.csv'

df_appdata = pd.read_csv(training_data_address, names=b)
df_label = pd.read_csv(training_label_address , names=['app_name','app_label'])
df_appdata = pd.merge(df_appdata, df_label, on=['app_name'])

#now, np.shape(result) = (100, 13628)
appdata_matrix = df_appdata.as_matrix()




#np.shape(X_train_ay) == (80,13628) #np.shape(X_test_ay) == (20, 13628)
# The first column of X_train_ay and X_test_ay is the app_name, and the last column is the app_label
m_all, n_all = np.shape(appdata_matrix)

X_data_matrix = df_appdata.as_matrix()

X_raw_all = X_data_matrix[:, 1: n_all - 1]
X_raw_all = X_raw_all.astype(float)
X_label_all = X_data_matrix[:, n_all - 1]
X_label_all = X_label_all.astype(str)


del df_appdata
gc.collect()


print('loading data finish!\n')
two = time.clock()
print('time of data_loading = %s \n'%(two - one))

print('train data SVD starts !\n')
U,s,V = LA.svd(X_raw_all, full_matrices=False)

#s_1000 = s[:1000]
#U_1000 = U[,:1000]
V_1000 = V[:1000,:]

del s
del U
del V
gc.collect()
print('test data SVD finishes !\n')
three = time.clock()
print('time of SVD = %s'%(three - two))
#V_address = 'C:/Users/Chaomin/Desktop/data_mining/data/all_V_1000.npy'
#V_1000 = np.load((V_address))

V_address = cwd + '/data/temp/all_V_1000'
np.save(V_address, V_1000)

three = time.clock()

print('time of SVD = %s \n'%(three- two))


four = time.clock()
print('Spliting data to test_data and training data:\n')
X_train_ay, X_test_ay, X_train_index, X_test_index = preprocessing.split(X_data_matrix, 0.2, 1)
#np.shape(X_train_ay) == (80,13628) #np.shape(X_test_ay) == (20, 13628)
# The first column of X_train_ay and X_test_ay is the app_name, and the last column is the app_label
m_test, n_test = np.shape(X_test_ay)
m_train, n_train = np.shape(X_train_ay)
X_test_label = X_test_ay[:, n_test - 1]
X_train_label = X_train_ay[:, n_train - 1]
X_test_appname = X_test_ay[:,0]
X_train_appname = X_train_ay[:,0]
X_raw_test = X_test_ay[:, 1:n_test - 1]
X_raw_train = X_train_ay[:, 1: n_train - 1]


X_raw_test_address = cwd + '/data/temp/X_raw_test'
np.save(X_raw_test_address, X_raw_test)

X_test_label_address = cwd + '/data/temp/X_test_label'
np.save(X_test_label_address, X_test_label)

#print(np.shape(X_raw_test))   (80,13626)
#print(np.shape(X_raw_train))  (20,13626)

X_raw_test = X_raw_test.astype(float)
X_raw_train = X_raw_train.astype(float)


X_train = np.dot(X_raw_train, V_1000.T)
y_train = X_train_label.astype(str)
y_test = X_test_label.astype(str)

five = time.clock()
print('Time of spliting data into train data and test data :%s \n'%(three- two))


for i in range(5):
    print('start {0}-th loop'.format(i + 1))
    time_mark_one = time.clock()

    label = np.unique(X_train_label)
    label = label.astype(str)

    number_of_fold = 5
    fold_index_list = preprocessing.k_fold_cv_withclass(number_of_fold, y_train)
    test_index, train_index = preprocessing.arrange_data_in_each_fold(fold_index_list,i, number_of_fold)



    print('{0}-th spliting data finishes!'.format(i + 1))
    print('time in split fold %s' % (four - three))
    X_in_train = X_train[train_index]
    X_in_test = X_train[test_index]
    y_in_train = y_train[train_index]
    y_in_test = y_train[test_index]

    m_in_test, n_in_test = np.shape(X_in_test)

    print('start training model {0}'.format(i+1))

    result_five_folds_dir = cwd + '/data/result/five_folds'
    if not os.path.exists( result_five_folds_dir):
        os.makedirs(result_five_folds_dir)

    matrix_withheader_address =  result_five_folds_dir + '/confusion_matrix_withheader' + str(i) + '.csv'
    matrix_address = result_five_folds_dir + '/confusion_matrix' +  str(i) + '.csv'
    acc_matrix_address = result_five_folds_dir + '/acc' +  str(i) + '.csv'

    result_weights_dir = cwd + '/data/result/weight_data'
    if not os.path.exists(result_weights_dir):
        os.makedirs(result_weights_dir)
    weights_address = result_weights_dir + '/weight' + str(i)



    logit_cl = lc.LogisticClassifier(label)
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

    time_mark_two = time.clock()
    print('Please check that: this is the {0}th round'.format(0 + 1))
    print('time in a loop %s seconds'%(time_mark_two - time_mark_one))

six = time.clock()
print('The time of 10 folds cross-validation:% s'%(six - one))
