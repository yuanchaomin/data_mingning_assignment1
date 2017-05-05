import numpy as np
import pandas as pd
import logistic_classifier as lc

def scale(X, with_mean = True, with_std = True, axis=0):
    np.asarray(X)
    X = X.astype('float')
    X[X == 0] = np.nan
    if with_mean:
        mean_ = np.mean(X, axis)
    if with_std:
        std_ = np.std(X,axis)
    X = (X - with_mean)/std_

    return X
def scale2(X, axis = 0):
    column_max = np.max(X, axis)
    column_min = np.min(X, axis)
    X = (X - column_min)/(column_max - column_min)

    return X

def split(X, test_data_size_ratio):
    """
     Split an data into two set, one is the index of training data set, and another is the  index oftest data set
     
    :param X: The data matrix with shape(m,n), where m is sample size and n is feature size 
    :param test_size: specify the size of test data
    :return: two array , one is the index of a training data, anther is  the index of test data 
    """
    m,n = np.shape(X)
    index = np.zeros((m,1))
    for i in range(m):
        index[i] = i
    X = np.concatenate([index,X], axis = 1)
    index = index.flatten()

    test_size = round(float(test_data_size_ratio) * len(index))
    X_test = np.random.choice(len(index), test_size, replace=False)

    X_df = pd.DataFrame(X)

    return X, X_test,X_df

if __name__ == "__main__":
    #a = np.matrix([[0,0,103],[40,50,6],[78,8,9]])
    data_address = 'C:/Users/Chaomin/Desktop/data_mining/data/classifier_data_for_test/test_data_for_logit_r/logit_train_data.csv'
    label_address = 'C:/Users/Chaomin/Desktop/data_mining/data/classifier_data_for_test/test_data_for_logit_r/logit_train_labelCopy.csv'
    predict_matrix_address = 'C:/Users/Chaomin/Desktop/data_mining/data/classifier_data_for_test/test_data_for_logit_r/logit_test_data.csv'
    data_loader = lc.DataLoader(data_address, label_address, predict_matrix_address)
    data, label, predict_matrix, unique_label = data_loader.return_value()

    q,w,e = split(data, 0.5)