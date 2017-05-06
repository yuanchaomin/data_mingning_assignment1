import pandas as pd
import numpy as np
import preprocessing
import eigen_ddcomposition as ed
import logistic_classifier
import compute_confusion_matrix2 as cc
import time

start_time = time.clock()
b = []
for i in range(1,13627):
    b.append('feature:'+ str(i))
b.insert(0,'app_name')

matrix_withheader_address = 'C:/Users/Chaomin/Desktop/data_mining/data/result/matrix_header.csv'
matrix_address = 'C:/Users/Chaomin/Desktop/data_mining/data/result/matrix.csv'

print('loading data begins !')
df_appdata = pd.read_csv('C:/Users/Chaomin/Desktop/data_mining/assignment1_2017S1/training_data.csv',names = b)
df_label =pd.read_csv('C:/Users/Chaomin/Desktop/data_mining/assignment1_2017S1/training_labels.csv', names = ['app_name','app_label'])
df_appdata = pd.merge(df_appdata, df_label, on = ['app_name'])
# now, np.shape(result) = (100, 13628)
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
#print(np.shape(X_raw_test))   (80,13626)
#print(np.shape(X_raw_train))  (20,13626)

df_train_appdata = df_appdata.ix[X_train_index]
df_train_mean = df_appdata.groupby('app_label').mean().reset_index()
#np.shape(df_train_mean) == (29, 13627), the first column is app_label, and the rest is 13627 features.
train_mean_mat = df_train_mean.iloc[:,1:]
train_mean_label = df_train_mean.iloc[:,0]
##np.shape(df_train_mean_mat) == (29, 13626), the app_label column was removed
##df_train_label is the label column in df_train_mean_mat
print('spliting data finish!\n')
print('PCA begins !\n')
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
df_svm.to_csv(svm_matrix_address,index=False)

preprocessing.clean_file(svm_matrix_address, cleaned_svm_matrix_address)
print('PCA finishes !\n')
print('Data transformation begins !\n')
## data transformation
svd_matrix =  np.genfromtxt('C:/Users/Chaomin/Desktop/data_mining/data/intermediate/cleaned_svd_matrix.csv', delimiter=',')

svd_matrix = np.matrix(svd_matrix[1:,:])
y_train = X_train_label
x_test = X_test_label
X_train = np.dot(X_raw_train,svd_matrix).astype(float)
X_test = np.dot(X_raw_test, svd_matrix).astype(float)
print('Data transformation finishes !\n')
print('Logistic model trainning starts !\n')
logit_cl = logistic_classifier.LogisticClassifier()


weights = logit_cl.fit(X_train, y_train, multiclass=True)
prob,in_matrix = logit_cl.predict_prob(X_test,weights,multiclass=True)

result_list = logit_cl.predict_label(prob,0.5, multiclass=True)

#label_address = 'C:/Users/Chaomin/Desktop/data_mining/data/intermediate/group_label.csv'
#label = np.genfromtxt(label_address, delimiter=',',dtype=str)
print('Logistic model trainning ends!\n')
print('Start to write final result!\n')
label = np.unique(X_train_label)
label = label.astype(str)

X_test_label = np.array(X_test_label, dtype=str)
result_array = np.array(result_list, dtype=str)
confusion_matrix_builder = cc.confusion_matrix_builder(X_test_label,result_array,label)
confusion_matrix, label_list = confusion_matrix_builder.compute_confusion_matrix()
acc_matrix = cc.calculate_r_c_sum(confusion_matrix)

row_header_l =list(['binary_classifier_by_label','TP','FN','FP','TN','ACC','SPE','PRE','F1'])
column_extra_string = 'binary_classifier_by_label'
result_df = cc.calculate_acc( confusion_matrix, acc_matrix,row_header_l,column_extra_string,label_list)
result_df.to_csv('C:/Users/Chaomin/Desktop/data_mining/data/result/acc_result3.csv', index = False)

print('final result writing finishes !\n')

end_time = time.clock()

print('Running time: %s Seconds'%(end-start))