# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 00:37:54 2017

@author: Chaomin
"""
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix

def r_c_sum(matrix, row, column):
    count = 0
    if column == 'all':
        for j in range(0, len(matrix)):
            if j != int(row):
                count += matrix[row, j]
    if row == 'all':
        for i in range(0, len(matrix)):
            if i != int(column):
                count += matrix[i, column]
    return count


def calculate_r_c_sum(matrix_arg):
    matrix = matrix_arg
    new_row = np.zeros((1,len(matrix)))
    new_column = np.empty([len(matrix) + 1, 1])
    matrix = np.concatenate([matrix,new_row], axis=0)
    matrix = np.concatenate([matrix,new_column], axis = 1)
    print(np.shape(matrix))
    print(np.shape(new_column))
    print(np.shape(new_row))
    print(matrix)
    for row in range(0,len(matrix)):
        matrix[row, len(matrix) - 1] = r_c_sum(matrix, row,'all')
    for column in range(0,len(matrix)):
        matrix[len(matrix) - 1, column] = r_c_sum(matrix, 'all', column)
    return matrix

class confusion_matrix_builder:
    def __init__(self, true_label_file_address, predicted_label_file_address, label_address):
        self.true_label_file_address = true_label_file_address
        self.predicted_label_file_address = predicted_label_file_address
        self.label_address = label_address

    def compute_confusion_matrix(self):
        true_label = np.genfromtxt(self.true_label_file_address, delimiter=',', dtype=str)
        predicted_label = np.genfromtxt(self.predicted_label_file_address, delimiter=',', dtype=str)
        label_list = list(np.genfromtxt(label_address, delimiter=',', dtype=str))
        confusion_matrix_ = confusion_matrix(true_label, predicted_label, labels=label_list)

        return confusion_matrix_, label_list

    def save_file(self, confusion_matrix, label_list, matrix_withheader_address, matrix_address):
        np.savetxt(matrix_address, confusion_matrix, fmt='%5d', delimiter=',')
        df_matrix = pd.DataFrame(confusion_matrix)
        df_matrix.columns = label_list
        df_matrix.insert(0, 'true_label||predicted_label', pd.Series(label_list))
        df_matrix.to_csv(matrix_withheader_address, index=False)

class analysis_confusion_matrix:
    def __init__(self, data_address):
        self.df = pd.read_csv(data_address)

    def calculate_binary_confusion_matix(self):
        rows = self.df.iterrows()
        return self.df



if __name__ == '__main__':
    true_label_file_address = 'C:/Users/Chaomin/Desktop/data_mining/data/train_test_data/my_test_label.csv'
    predict_label_file_address = 'C:/Users/Chaomin/Desktop/data_mining/data/result/logit_test_label.csv'
    label_address = 'C:/Users/Chaomin/Desktop/data_mining/data/intermediate/group_label.csv'
    matrix_withheader_address = 'C:/Users/Chaomin/Desktop/data_mining/data/result/matrix_header.csv'
    matrix_address = 'C:/Users/Chaomin/Desktop/data_mining/data/result/matrix.csv'
    confusion_matrix_address = 'C:/Users/Chaomin/Desktop/data_mining/data/result/matrix_header.csv'
    df = analysis_confusion_matrix(confusion_matrix_address).calculate_binary_confusion_matix()
    k = confusion_matrix_builder(true_label_file_address,predict_label_file_address,label_address)
    j, l = k.compute_confusion_matrix()
    # a = np.matrix('1 2; 3 4')
    # print(r_c_sum(a, 1, 'all'))
    # print(r_c_sum(a, 0, 'all'))
    # print(r_c_sum(a, 'all', 0))
    # print(r_c_sum(a, 'all', 1))
    j_2 = calculate_r_c_sum(j)
    pd.DataFrame(j_2).to_csv('C:/Users/Chaomin/Desktop/data_mining/data/result/test_sum.csv')
