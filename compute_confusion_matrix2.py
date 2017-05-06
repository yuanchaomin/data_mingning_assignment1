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
    new_column = np.zeros((len(matrix) + 1, 1))
    matrix = np.concatenate([matrix,new_row], axis=0)
    matrix = np.concatenate([matrix,new_column], axis = 1)
    for row in range(0,len(matrix)):
        matrix[row, len(matrix) - 1] = r_c_sum(matrix, row,'all')
    for column in range(0,len(matrix)):
        matrix[len(matrix) - 1, column] = r_c_sum(matrix, 'all', column)
    return matrix

def calculate_acc(matrix, intermediate_matrix, row_header_l, column_extra_string, label_l, data_sample_size):
    matrix_len = len(matrix)
    result_matrix = np.zeros((matrix_len, len(row_header_l)))
    row_header = np.matrix(row_header_l)
    result_matrix = np.concatenate([row_header, result_matrix], axis=0)
    label_l.insert(0, column_extra_string)
    column_header = np.matrix(label_l).T
    result_matrix = np.concatenate([column_header, result_matrix], axis=1)
    result_matrix = np.delete(result_matrix, 1, 1)
    result_df = pd.DataFrame(result_matrix, columns=row_header_l)
    result_sub_df = result_df.iloc[1:, 1:len(row_header_l)]

    for i in range(0,matrix_len):
        result_sub_df.set_value(i + 1,'TP',intermediate_matrix[i,i])
        result_sub_df.set_value(i + 1, 'FN', intermediate_matrix[matrix_len,i])
        result_sub_df.set_value(i + 1, 'FP', intermediate_matrix[i,matrix_len])
        result_sub_df.set_value(i + 1, 'TN', int(data_sample_size)-(intermediate_matrix[i,i] + intermediate_matrix[matrix_len,i] + intermediate_matrix[i,matrix_len]))

    for i in range(0,matrix_len):
        TP = result_sub_df.get_value(i+1,'TP')
        TN = result_sub_df.get_value(i+1,'TN')
        FN = result_sub_df.get_value(i+1,'FN')
        FP = result_sub_df.get_value(i+1,'FP')

        acc = (TP+TN)/(TP+TN+FN+FP)
        spe = TN/(TN + FN)
        pre = TP/(TP+FP)
        f1 =  2*TP/(2*TP + FP + FN)

        result_sub_df.set_value(i + 1, 'ACC', acc)
        result_sub_df.set_value(i + 1, 'SPE', spe)
        result_sub_df.set_value(i + 1, 'PRE', pre)
        result_sub_df.set_value(i + 1, 'F1', f1)
    del label_l[0]
    result_sub_df.insert(0, 'binary_classifier_by_label', label_l)

    return result_sub_df

class confusion_matrix_builder:
    def __init__(self, true_label, predicted_label, label):
        self.true_label = true_label
        self.predicted_label = predicted_label
        self.unique_label = label

    def compute_confusion_matrix(self):
        true_label = self.true_label
        predicted_label = self.predicted_label
        label_list = list(self.unique_label)
        confusion_matrix_ = confusion_matrix(true_label, predicted_label, labels=label_list)

        return confusion_matrix_, label_list

    def save_file(self, confusion_matrix, label_list, matrix_withheader_address, matrix_address):
        np.savetxt(matrix_address, confusion_matrix, fmt='%5d', delimiter=',')
        df_matrix = pd.DataFrame(confusion_matrix)
        df_matrix.columns = label_list
        df_matrix.insert(0, 'true_label||predicted_label', pd.Series(label_list))
        df_matrix.to_csv(matrix_withheader_address, index=False)


if __name__ == '__main__':
    true_label_file_address = 'C:/Users/Chaomin/Desktop/data_mining/data/train_test_data/my_test_label.csv'
    predict_label_file_address = 'C:/Users/Chaomin/Desktop/data_mining/data/result/logit_test_label.csv'
    label_address = 'C:/Users/Chaomin/Desktop/data_mining/data/intermediate/group_label.csv'
    true_label =  np.genfromtxt(true_label_file_address, delimiter=',',dtype=str)
    predict_label = np.genfromtxt(predict_label_file_address, delimiter=',', dtype=str)
    label = np.genfromtxt(label_address, delimiter=',', dtype=str)
    #matrix_withheader_address = 'C:/Users/Chaomin/Desktop/data_mining/data/result/matrix_header.csv'
    #matrix_address = 'C:/Users/Chaomin/Desktop/data_mining/data/result/matrix.csv'
    #confusion_matrix_address = 'C:/Users/Chaomin/Desktop/data_mining/data/result/matrix_header.csv'
    row_header_l =list(['binary_classifier_by_label','TP','FN','FP','TN','ACC','SPE','PRE','F1'])
    column_extra_string = 'binary_classifier_by_label'

    k = confusion_matrix_builder(true_label,predict_label,label)
    j, l = k.compute_confusion_matrix()
    j_2 = calculate_r_c_sum(j)

    result_df = calculate_acc(j, j_2,row_header_l,column_extra_string,l)
    result_df.to_csv('C:/Users/Chaomin/Desktop/data_mining/data/result/acc_result2.csv', index = False)