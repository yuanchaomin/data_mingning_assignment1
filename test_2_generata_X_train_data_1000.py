import pandas as pd
import numpy as np
from sklearn import linear_model as lm
import compute_confusion_matrix2 as cc
import logistic_classifier as lc
import time
import importlib
s_1000 = np.load('C:/Users/Chaomin/Desktop/data_mining/data/intermediate/my_s_1000_np_float64.npy')
U_1000 = np.load('C:/Users/Chaomin/Desktop/data_mining/data/intermediate/my_U_1000_np_float64.npy')
V_1000 = np.load('C:/Users/Chaomin/Desktop/data_mining/data/intermediate/my_V_1000_np_float64.npy')

X_train_df = pd.read_csv('C:/Users/Chaomin/Desktop/data_mining/data/intermediate/X_train.csv')
X_test_df = pd.read_csv('C:/Users/Chaomin/Desktop/data_mining/data/intermediate/X_test.csv')


X_train_mat = X_train_df.as_matrix()
X_test_mat = X_test_df.as_matrix()


m_train, n_train = np.shape(X_train_mat)
m_test, n_test = np.shape(X_test_mat)


X_train_label = X_train_mat[:,n_train - 1]
X_train_label = X_train_label.astype(str)
X_test_label = X_test_mat[:,n_test - 1]
X_test_label = X_test_label.astype(str)


X_raw_train = X_train_mat[:, 1: n_train - 1]
X_raw_train = X_raw_train.astype(float)
X_raw_test = X_test_mat[:, 1: n_train - 1]
X_raw_test = X_raw_test.astype(float)

X_train = np.dot(X_raw_train, V_1000.T)
X_test = np.dot(X_raw_test, V_1000.T)
y_train = X_train_label
y_test = X_test_label


# logit_classifier = lm.LogisticRegression(multi_class= 'ovr', n_jobs= -1)
# logit_classifier.fit(X_train,y_train)
# print(logit_classifier.coef_)
# print(logit_classifier.intercept_)
importlib.reload(lc)
first_time_location = time.clock()
logit_cl = lc.LogisticClassifier()
weights = logit_cl.fit(X_train, y_train, multiclass=True)
prob,in_matrix = logit_cl.predict_prob(X_test,weights,multiclass=True)
result_list = logit_cl.predict_label(prob,0.5, multiclass=True)
predict_result_array = np.array(result_list, dtype=str)
# predict_result_array = logit_classifier.predict(X_test)

matrix_withheader_address = 'C:/Users/Chaomin/Desktop/data_mining/data/intermediate/result/matrix_header_sk6.csv'
matrix_address = 'C:/Users/Chaomin/Desktop/data_mining/data/intermediate/result/matrix_sk6.csv'

label_address = 'C:/Users/Chaomin/Desktop/data_mining/data/intermediate/try_y_train.csv'
label = np.genfromtxt(label_address, delimiter=',', dtype=str)
label = label[1:]
label = np.unique(label)
#np.save('C:/Users/Chaomin/Desktop/data_mining/data/intermediate/unique_label_ndarray',label)

confusion_matrix_builder = cc.confusion_matrix_builder(y_test,predict_result_array,label)
confusion_matrix, label_list = confusion_matrix_builder.compute_confusion_matrix()
confusion_matrix_builder.save_file(confusion_matrix,label_list, matrix_withheader_address,matrix_address)
acc_matrix = cc.calculate_r_c_sum(confusion_matrix)

row_header_l =list(['binary_classifier_by_label','TP','FN','FP','TN','ACC','SPE','PRE','F1'])
column_extra_string = 'binary_classifier_by_label'
result_df = cc.calculate_acc( confusion_matrix, acc_matrix,row_header_l,column_extra_string,label_list, data_sample_size=m_test)
#result_df.to_csv('C:/Users/Chaomin/Desktop/data_mining/data/intermediate/result/acc.csv', index=False)
result_df.to_csv('C:/Users/Chaomin/Desktop/data_mining/data/intermediate/result/acc6.csv', index=False)
second_time_location = time.clock()

print('Running time: %s Seconds'%(second_time_location-first_time_location))