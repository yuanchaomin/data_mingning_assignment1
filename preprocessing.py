import numpy as np
import pandas as pd
import re
import logistic_classifier as lc
import random


def scale(X, with_mean=True, with_std=True, axis=0):
    np.asarray(X)
    X = X.astype('float')
    X[X == 0] = np.nan
    if with_mean:
        mean_ = np.mean(X, axis)
    if with_std:
        std_ = np.std(X, axis)
    X = (X - with_mean) / std_

    return X


def scale2(X, axis=0):
    column_max = np.max(X, axis)
    column_min = np.min(X, axis)
    X = (X - column_min) / (column_max - column_min)

    return X


def split(X, test_data_size_ratio, random_state_number):
    """
     Split an data into two set, one is the index of training data set, and another is the  index oftest data set

    :param X: The data matrix with shape(m,n), where m is sample size and n is feature size 
    :param test_size: specify the size of test data
    :param random_state_number: the random state number
    :return: two array , one is the index of a training data, anther is  the index of test data 
    """
    random_state = np.random.RandomState(random_state_number)

    X_df = pd.DataFrame(X)
    index = X_df.index
    # print(type(index))
    test_size = round(float(test_data_size_ratio) * len(index))
    X_test_index = random_state.choice(len(index), test_size, replace=False)
    # print(type(X_test_index))
    X_train_index = list(set(index) - set(X_test_index))

    X_test_df = X_df.ix[X_test_index]
    X_train_df = X_df.ix[X_train_index]

    X_test_ay = X_test_df.as_matrix()
    X_train_ay = X_train_df.as_matrix()

    return X_train_ay, X_test_ay, X_train_index, X_test_index


def clean_file(input_file_address, out_file_address):
    with open(out_file_address, 'w') as writefile:
        with open(input_file_address, 'r') as readfile:
            for line in readfile.readlines():
                newline = re.sub(r'\(|\)|\+0.{18}0e\+00j|\+0j|j', '', line)
                writefile.write(newline)
                # print(line)
            writefile.close()
            readfile.close()


def k_fold_cv(k, n_row):
    folds_ = range(0, n_row - 1)
    random.shuffle(folds_)
    step = int(np.ceil(n_row / float(k)))
    folds_ = [folds_[i:(i + step)] for i in range(0, n_row, step)]
    return folds_


def k_fold_cv_withclass(k, label):
    """This is UDF for k-fold"""
    # initilisation
    dict_folds_ = {}
    special_list = []
    label_df = pd.DataFrame(label)
    label_df.columns = ['label']
    label_name = np.unique(label_df['label'])
    dict_name_index = {}
    # put indices under different labels
    # curr_name = each label name
    # the element under curr_name  = the index
    for curr_name in label_name:
        dict_name_index[curr_name] = [i for i in label_df[label_df['label'] == curr_name].index]
    # split the indices based on k-fold
    for name, index in dict_name_index.items():
        # shuffle the index
        random.seed(1)
        random.shuffle(index)
        # the length of label
        label_row = len(index)
        # step  = the length for each fold. e.g. numer of label is 100, k = 10, then step = 10.
        step = int(np.ceil(label_row / float(k)))
        # build a folds_list for each label
        each_name_folds_ = [index[i:(i + step)] for i in range(0, label_row, step)]
        # then we put each index under different lable into same ten-fold
        # in this way, even if eachf_name_folds < 10, it does not matter
        for sublist_index in range(len(each_name_folds_)):
            if sublist_index not in dict_folds_:
                dict_folds_[sublist_index] = []
            dict_folds_[sublist_index].extend(each_name_folds_[sublist_index])
    # convert dict into list. e.g. something like, [[1,2,3], [1,2,3]......] - there are k sublist
    dict_folds_ = [mysublist for index, mysublist in dict_folds_.items()]

    return dict_folds_


def arrange_data_in_each_fold(result_list, selected_test_fold_index, fold_size):
    infold_index = [i for i in range(fold_size)]
    test_index = result_list[infold_index[selected_test_fold_index]]
    infold_test_index = []
    for i in infold_index:
        if i != selected_test_fold_index:
            infold_test_index.append(i)

    train_index = [result_list[i] for i in infold_test_index]

    train_final_index = []
    for i in train_index:
        for j in i:
            train_final_index.append(j)
    return test_index, train_final_index
    # return test_index_ay


# if __name__ == "__main__":
#     #a = np.matrix([[0,0,103],[40,50,6],[78,8,9]])
#     data_address = 'C:/Users/Chaomin/Desktop/data_mining/data/classifier_data_for_test/test_data_for_logit_r/logit_train_data.csv'
#     label_address = 'C:/Users/Chaomin/Desktop/data_mining/data/classifier_data_for_test/test_data_for_logit_r/logit_train_labelCopy.csv'
#     predict_matrix_address = 'C:/Users/Chaomin/Desktop/data_mining/data/classifier_data_for_test/test_data_for_logit_r/logit_test_data.csv'
#     data_loader = lc.DataLoader(data_address, label_address, predict_matrix_address)
#     data, label, predict_matrix, unique_label = data_loader.return_value()
#
#     X_train_ay, X_test_ay = split(data, 0.2)

if __name__ == "__main__":
    # a = np.ones((100,))
    # a = a.flatten()
    file = pd.read_csv('C:/Users/Chaomin/Desktop/data_mining/data/train_test_data/my_test_label.csv')
    file_ay = np.array(file, dtype=str)
    index = file.index
    result = k_fold_cv_withclass(10, file_ay)
    j, k = arrange_data_in_each_fold(result, 0, 10)
