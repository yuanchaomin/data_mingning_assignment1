import pandas as pd
import numpy as np
import logistic_classifier as lc
import os

cwd = os.getcwd()

test_data_address = cwd + '/data/test_data.csv'
V_address = cwd + '/data/temp/all_V_1000.npy'
#V_address = 'C:/Users/Chaomin/Desktop/data_mining/data/all_V_1000.npy'
result_address = cwd + '/data/result/predict_result.csv'
unique_label_address = cwd + '/data/temp/unique_label.npy'
weights_address = cwd + '/data/result/weight_data/weight' + str(0) +'.npy'

df_test_data = pd.read_csv(test_data_address)

test_data = df_test_data.as_matrix()

test_data_appname = test_data[:,0]
test_data_appname = test_data_appname.astype(str)
test_raw_data = test_data[:, 1:]
test_raw_data = test_raw_data.astype(float)

V_1000 = np.load(V_address)

X_test = np.dot(test_raw_data, V_1000.T)

unique_label = np.load(unique_label_address)

logit_cl = lc.LogisticClassifier(unique_label)
weights = np.load(weights_address)
prob, in_matrix = logit_cl.predict_prob(X_test, weights, multiclass=True)
result_list = logit_cl.predict_label(prob, 0.5, multiclass=True)

df_result = pd.DataFrame(test_data_appname, columns = ['app_name'])
df_result['label'] = result_list

df_result.to_csv(result_address, index=False)





