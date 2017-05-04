import numpy as np
from sklearn import preprocessing

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
a = np.matrix([[0,0,103],[40,50,6],[78,8,9]])

b = preprocessing.scale(a)
print(b)

