import pandas as pd
import numpy as np
import preprocessing
from numpy import linalg as LA
import logistic_classifier as lc
import eigen_ddcomposition as ed
import logistic_classifier
import compute_confusion_matrix2 as cc
import time

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

print('loading data finish!\n')
print('Spliting data to test_data and training data:')
X_train_ay, X_test_ay, X_train_index, X_test_index = preprocessing.split(appdata_matrix, 0.2)
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

X_raw_test = X_raw_test.astype(np.float16)
X_raw_train = X_raw_train.astype(np.float16)
#print(np.shape(X_raw_test))   (80,13626)
#print(np.shape(X_raw_train))  (20,13626)
U,s,V = LA.svd(X_raw_train, full_matrices=False)

s_1000 = s[:1000]
U_1000 = U[,:1000]
V_1000 = V[:1000,:]

X_train_all = np.dot(X_raw_train, V_1000.T)
X_test_all = np.dot(X_raw_test, V_1000.T)


label = np.unique(X_train_label)
label = label.astype(str)

number_of_fold = 10
fold_index_list = preprocessing.k_fold_cv_withclass(number_of_fold, X_raw_train)
for i in range(number_of_fold):
        test_index, train_index = preprocessing.arrange_data_in_each_fold(fold_index_list,i,number_of_fold)
        X_train = X_train_all[train_index]
        X_test = X_test_all[test_index]
        y_train = X_train_label[train_index]
        y_test = X_test_label[test_index]

        logit_cl = lc.LogisticClassifier()
        weights = logit_cl.fit(X_train, y_train, multiclass=True)
        prob, in_matrix = logit_cl.predict_prob(X_test, weights, multiclass=True)
        result_list = logit_cl.predict_label(prob, 0.5, multiclass=True)
        predict_result_array = np.array(result_list, dtype=str)

        matrix_withheader_address = 'C:/Users/Chaomin/Desktop/data_mining/data/intermediate/result/win/confusion_matrix_withheader' + str(i) + '.csv'
        matrix_address = 'C:/Users/Chaomin/Desktop/data_mining/data/intermediate/result/win/confusion_matrix' + str(i) + '.csv'
        acc_matrix_address = 'C:/Users/Chaomin/Desktop/data_mining/data/intermediate/result/win/acc' + str(i) + '.csv'
        confusion_matrix_builder = cc.confusion_matrix_builder(y_test, predict_result_array, label)
        confusion_matrix, label_list = confusion_matrix_builder.compute_confusion_matrix()
        confusion_matrix_builder.save_file(confusion_matrix, label_list, matrix_withheader_address, matrix_address)
        acc_matrix = cc.calculate_r_c_sum(confusion_matrix)

        row_header_l = list(['binary_classifier_by_label', 'TP', 'FN', 'FP', 'TN', 'ACC', 'SPE', 'PRE', 'F1'])
        column_extra_string = 'binary_classifier_by_label'
        result_df = cc.calculate_acc(confusion_matrix, acc_matrix, row_header_l, column_extra_string, label_list,
                                     data_sample_size=m_test)
        # result_df.to_csv('C:/Users/Chaomin/Desktop/data_mining/data/intermediate/result/acc.csv', index=False)

        result_df.to_csv(acc_matrix_address, index=False)

