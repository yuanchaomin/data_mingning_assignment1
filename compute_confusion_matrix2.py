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
    # print(np.shape(matrix))
    # print(np.shape(new_column))
    # print(np.shape(new_row))
    # print(matrix)
    for row in range(0,len(matrix)):
        matrix[row, len(matrix) - 1] = r_c_sum(matrix, row,'all')
    for column in range(0,len(matrix)):
        matrix[len(matrix) - 1, column] = r_c_sum(matrix, 'all', column)
    return matrix

def calculate_acc(matrix, row_header_l, column_header_l, label_l):
    matrix_len = len(matrix)
    result_matrix = np.zeros((matrix_len, len(row_header_l)))
    row_header = np.matrix(row_header_l)
    result_matrix = np.concatenate([row_header, result_matrix], axis=0)
    l.insert(0, 'binary_classifier_by_label')
    column_header = np.matrix(l).T
    result_matrix = np.concatenate([column_header, result_matrix], axis=1)
    result_matrix = np.delete(result_matrix, 1, 1)
    result_df = pd.DataFrame(result_matrix, columns=row_header_l)
    result_sub_df = result_df.iloc[1:, 1:len(row_header_l)]

    for i in range(0,matrix_len):
        result_sub_df.set_value(i + 1,'TP',j_2[i,i])
        result_sub_df.set_value(i + 1, 'FN', j_2[matrix_len,i])
        result_sub_df.set_value(i + 1, 'FP', j_2[i,matrix_len])
        result_sub_df.set_value(i + 1, 'TN', np.sum(j_2)-(j_2[i,i] + j_2[matrix_len,i] + j_2[i,matrix_len]))

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
    del l[0]
    result_sub_df.insert(0, 'binary_classifier_by_label', l)

    return result_sub_df



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
    #pd.DataFrame(j_2).to_csv('C:/Users/Chaomin/Desktop/data_mining/data/result/test_sum.csv')

    # result_matrix = np.zeros((len(j), 9))
    # row_header = np.matrix(['binary_classifier_by_label','TP','FN','FP','TN','ACC','SPE','PRE','F1'])
    # result_matrix = np.concatenate([row_header, result_matrix], axis = 0)
    # l.insert(0, 'binary_classifier_by_label')
    # column_header = np.matrix(l).T
    # result_matrix = np.concatenate([column_header,result_matrix], axis = 1)
    # result_matrix = np.delete(result_matrix, 1,1)
    # print('result_matrix',np.shape(result_matrix))
    # print('row_header', np.shape(row_header))
    # print('column_header',np.shape(column_header))
    #
    # #print(result_matrix)
    # result_df = pd.DataFrame(result_matrix, columns=['binary_classifier_by_label','TP','FN','FP','TN','ACC','SPE','PRE','F1'])
    # result_sub_df = result_df.iloc[1:,1:9]
    # print(np.shape(result_sub_df))
    # #result_sub_df
    # for i in range(0,30):
    #     result_sub_df.set_value(i + 1,'TP',j_2[i,i])
    #     result_sub_df.set_value(i + 1, 'FN', j_2[30,i])
    #     result_sub_df.set_value(i + 1, 'FP', j_2[i,30])
    #     result_sub_df.set_value(i + 1, 'TN', np.sum(j_2)-(j_2[i,i] + j_2[30,i] + j_2[i,30]))
    #
    # for i in range(0,30):
    #     TP = result_sub_df.get_value(i+1,'TP')
    #     TN = result_sub_df.get_value(i+1,'TN')
    #     FN = result_sub_df.get_value(i+1,'FN')
    #     FP = result_sub_df.get_value(i+1,'FP')
    #
    #     acc = (TP+TN)/(TP+TN+FN+FP)
    #     spe = TN/(TN + FN)
    #     pre = TP/(TP+FP)
    #     f1 =  2*TP/(2*TP + FP + FN)
    #     result_sub_df.set_value(i + 1, 'ACC', acc)
    #     result_sub_df.set_value(i + 1, 'SPE', spe)
    #     result_sub_df.set_value(i + 1, 'PRE', pre)
    #     result_sub_df.set_value(i + 1, 'F1', f1)
    #
    # del l[0]
    # result_sub_df.insert(0, 'binary_classifier_by_label', l)