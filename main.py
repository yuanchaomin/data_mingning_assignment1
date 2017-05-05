import pandas as pd
import numpy as np
import preprocessing
import eigen_ddcomposition as ed
b = []
for i in range(1,13627):
    b.append('feature:'+ str(i))
b.insert(0,'app_name')

df_appdata = pd.read_csv('C:/Users/Chaomin/Desktop/data_mining/assignment1_2017S1/training_data.csv',names = b, nrows =100)
df_label =pd.read_csv('C:/Users/Chaomin/Desktop/data_mining/assignment1_2017S1/training_labels.csv', names = ['app_name','app_label'])
df_appdata = pd.merge(df_appdata, df_label, on = ['app_name'])
# now, np.shape(result) = (100, 13628)
appdata_matrix = df_appdata.as_matrix()

X_train_ay, X_test_ay, X_train_index, X_test_index = preprocessing.split(appdata_matrix, 0.2)
#np.shape(X_train_ay) == (80,13628) #np.shape(X_test_ay) == (20, 13628)
# The first column of X_train_ay and X_test_ay is the app_name, and the last column is the app_label
m_test, n_test = np.shape(X_test_ay)
X__test_appname = X_test_ay[:, n_test - 1]
X_test_label = X_test_ay[:,0]
X_test = X_test_ay[:, 1:n_test - 1]
#print(np.shape(X_test))   (80,13626)

df_train_appdata = df_appdata.ix[X_train_index]
df_train_mean = df_appdata.groupby('app_label').mean().reset_index()
#np.shape(df_train_mean) == (29, 13627), the first column is app_label, and the rest is 13627 features.
train_mean_mat = df_train_mean.iloc[:,1:]
train_mean_label = df_train_mean.iloc[:,0]
#np.shape(df_train_mean_mat) == (29, 13626), the app_label column was removed
#df_train_label is the label column in df_train_mean_mat

eigend_values, eigend_vectors, c_m = ed.eigend(train_mean_mat)
svd_dict = dict(zip(eigend_values, eigend_vectors))
start = 0.001
end = 100
filtered_svd_dict = ed.filter_eigenv(svd_dict, float(start), float(end))
final_e_l = list(i for i in sorted(filtered_svd_dict.keys(),reverse=True))
final_v_l = list(filtered_svd_dict[i] for i in final_e_l)
final_e_array = np.array(final_e_l)
final_v_matrix = np.matrix(final_v_l).T

svm_matrix_address = 'C:/Users/Chaomin/Desktop/data_mining/data/intermediate/svd_matrix.csv'
cleaned_svm_matrix_address = 'C:/Users/Chaomin/Desktop/data_mining/data/intermediate/cleaned_svd_matrix.csv'
df_svm = pd.DataFrame(final_v_matrix)
df_svm.to_csv(svm_matrix_address,index = False)

preprocessing.clean_file(svm_matrix_address, cleaned_svm_matrix_address)



