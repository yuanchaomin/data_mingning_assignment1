import pandas as pd
import numpy as np
from numpy import linalg as LA
import time
import preprocessing

# b = []
# for i in range(1,13627):
#     b.append('feature:'+ str(i))
# b.insert(0,'app_name')
#
# first_time_location = time.clock()
# print('loading data begins !')
# df_appdata = pd.read_csv('C:/Users/Chaomin/Desktop/data_mining/assignment1_2017S1/training_data.csv', names = b)
# df_label =pd.read_csv('C:/Users/Chaomin/Desktop/data_mining/assignment1_2017S1/training_labels.csv', names = ['app_name','app_label'])
# df_appdata = pd.merge(df_appdata, df_label, on = ['app_name'])
# #now, np.shape(result) = (100, 13628)
# appdata_matrix = df_appdata.as_matrix()
#
# print('loading data finish!\n')
#
#
# print('Spliting data to test_data and training data:')
# X_train_ay, X_test_ay, X_train_index, X_test_index = preprocessing.split(appdata_matrix, 0.2)
# #np.shape(X_train_ay) == (80,13628) #np.shape(X_test_ay) == (20, 13628)
# # The first column of X_train_ay and X_test_ay is the app_name, and the last column is the app_label
# m_test, n_test = np.shape(X_test_ay)
# m_train, n_train = np.shape(X_train_ay)
# X_test_label = X_test_ay[:, n_test - 1]
# X_train_label = X_train_ay[:,n_train - 1]
# X_test_appname = X_test_ay[:,0]
# X_train_appname = X_train_ay[:,0]
# X_raw_test = X_test_ay[:, 1:n_test - 1]
# X_raw_train = X_train_ay[:, 1: n_train - 1]
#
# df_X_train = pd.DataFrame(X_train_ay)
# df_X_test = pd.DataFrame(X_test_ay)
# df_X_train_index = pd.DataFrame(X_train_index)
# df_X_test_index = pd.DataFrame(X_test_index)
#
# df_X_train.to_csv('C:/Users/Chaomin/Desktop/data_mining/data/intermediate/X_train.csv', index=False)
# df_X_test.to_csv('C:/Users/Chaomin/Desktop/data_mining/data/intermediate/X_test.csv', index=False)
# df_X_train_index.to_csv('C:/Users/Chaomin/Desktop/data_mining/data/intermediate/X_train_index.csv', index=False)
# df_X_test_index.to_csv('C:/Users/Chaomin/Desktop/data_mining/data/intermediate/X_test_index.csv', index=False)
# second_time_location = time.clock()
#
# print('loading and spliting Running time: %s Seconds \n'%(second_time_location-first_time_location))
first_time_location = time.clock()
X_train_df = pd.read_csv('C:/Users/Chaomin/Desktop/data_mining/data/intermediate/X_train.csv')
#np.shape(X_train_df) == (16803, 13628)
X_train_mat = X_train_df.as_matrix()
#np.shape(X_train_mat) == (16803, 13628), ndarray, dtype=object remove (assumed number header)
m_train, n_train = np.shape(X_train_mat)
X_train_label = X_train_mat[:,n_train - 1]
X_train_label = X_train_label.astype(str)
#
#np.shape(X_train_label) == (16083,)  dtype = '<U19'
X_raw_train = X_train_mat[:, 1: n_train - 1]
X_raw_train = X_raw_train.astype(float)
#np.shape(X_raw_train) == (16083, 13626) dtype = 'float'
print('Now perform an SVD\n')
third_time_location = time.clock()
U,s,V = LA.svd(X_raw_train, full4_matrices=False)
fourth_time_location = time.clock()
print('SVD time is: %s Seconds \n'%(fourth_time_location-third_time_location))
