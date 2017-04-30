# -*- coding: utf-8 -*-
"""
Created on Sun Apr 30 00:37:54 2017

@author: Chaomin
"""
import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix


def compute_confusion_matrix(true_label_file, predicted_label_file,label_address):
     true_label = np.genfromtxt(true_label_file,delimiter = ',', dtype= str)
     predicted_label = np.genfromtxt(predicted_label_file, delimiter = ',', dtype= str)
     label_list = list(np.genfromtxt(label_address, delimiter = ',',dtype= str))
     confusion_matrix_ = confusion_matrix(true_label, predicted_label, labels = label_list)
     
     return confusion_matrix_, label_list
 

true_label_file_address = 'C:/Users/Chaomin/Desktop/new/data_mining/data/train_test_data/my_test_label.csv'
predict_label_file_address = 'C:/Users/Chaomin/Desktop/new/data_mining/data/result/logit_test_label.csv'
label_address = 'C:/Users/Chaomin/Desktop/new/data_mining/data/intermieidate/group_label.csv'

class confusion_matrix_builder:
    def __init__(self,true_label_file_address, predicted_label_file_address, label_address):
        self.true_label_file_address = true_label_file_address
        self.predicted_label_file_address = predicted_label_file_address
        self.label_address = label_address
    
    def compute_confusion_matrix(self):
        true_label = np.genfromtxt(self.true_label_file_address, delimiter = ',',dtype=str)
        predicted_label = np.genfromtxt(self.predicted_label_file_address, delimiter = ',', dtype= str)
        label_list = list(np.genfromtxt(label_address, delimiter = ',',dtype= str))
        confusion_matrix_ = confusion_matrix(true_label, predicted_label, labels = label_list)

        return confusion_matrix_, label_list
    


    def save_file(self, confusion_matrix, label_list, matrix_withheader_address, matrix_address):
        np.savetxt(matrix_address, confusion_matrix, fmt='%5d', delimiter=',')
        df_matrix = pd.DataFrame(confusion_matrix)
        df_matrix.columns = label_list
        df_matrix.insert(0,'true_label||predicted_label', pd.Series(label_list))
        df_matrix.to_csv(matrix_withheader_address, index = False)

# if __name__ == '__main__':
#     true_label_file_address = 'C:/Users/Chaomin/Desktop/new/data_mining/data/train_test_data/my_test_label.csv'
#     predict_label_file_address = 'C:/Users/Chaomin/Desktop/new/data_mining/data/result/logit_test_label.csv'
#     label_address = 'C:/Users/Chaomin/Desktop/new/data_mining/data/intermieidate/group_label.csv'
#     matrix_withheader_address = 'C:/Users/Chaomin/Desktop/new/data_mining/data/result/matrix_header.csv'
#     matrix_address = 'C:/Users/Chaomin/Desktop/new/data_mining/data/result/matrix.csv'
#
#     k = confusion_matrix_builder(true_label_file_address,predict_label_file_address,label_address)
#     m,n = k.compute_confusion_matrix()
#     k.save_file(m,n,matrix_withheader_address,matrix_address)


class  analysis_confusion_matrix:
    def __init__(self, data_address):
        self.df = pd.read_csv(data_address)

    def calculate_binary_confusion_matix(self):
        rows = self.df.iterrows()
        return self.df



if __name__ == '__main__':
    true_label_file_address = 'C:/Users/Chaomin/Desktop/new/data_mining/data/train_test_data/my_test_label.csv'
    predict_label_file_address = 'C:/Users/Chaomin/Desktop/new/data_mining/data/result/logit_test_label.csv'
    label_address = 'C:/Users/Chaomin/Desktop/new/data_mining/data/intermieidate/group_label.csv'
    matrix_withheader_address = 'C:/Users/Chaomin/Desktop/new/data_mining/data/result/matrix_header.csv'
    matrix_address = 'C:/Users/Chaomin/Desktop/new/data_mining/data/result/matrix.csv'
    confusion_matrix_address = 'C:/Users/Chaomin/Desktop/new/data_mining/data/result/matrix_header.csv'
    df = analysis_confusion_matrix(confusion_matrix_address).calculate_binary_confusion_matix()
    j,l = compute_confusion_matrix(true_label_file_address,predict_label_file_address,label_address)

    l.insert(0,'true_label||predicted_label')
    l2 = []
    for j in l:
        if j != l[0]:
            l2.append(j)
    l3 = []
    for i in range(0,len(l2)):
        l3.append(l[0])

    index_zip = zip(l3,l2)

    index_list =[]
    for i,j in index_zip:
        index_list.append(df[[str(i), str(j)]])


######