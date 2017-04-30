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
 

true_label_file_address = 'C:/Users/Chaomin/Desktop/new/data_mining/data/train_test_data'
predict_label_file_address = 'C:/Users/Chaomin/Desktop/new/data_mining/data/result/logit_test_label.csv'
label_address = 'C:/Users/Chaomin/Desktop/new/data_mining/data/intermieidate/group_label.csv'

#==============================================================================
# matrix, label_list = compute_confusion_matrix(true_label_file_address, predict_label_file_address, label_address)
# np.savetxt('test.csv',  matrix,fmt='%5d', delimiter=',') 
# 
# df_matrix = pd.DataFrame(matrix)
# #label_list.insert(0,'true_label/predicted_label')
# df_matrix.columns = label_list
# df_matrix.insert(0,'true_label||predicted_label', pd.Series(label_list))
# df_matrix.to_csv('true_table2.csv', index = False)
# 
# 
#==============================================================================
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

if __name__ == '__main__':
    k = confusion_matrix_builder(true_label_file_address,predict_label_file_address,label_address)
    #m,n = k.compute_confusion_matrix(False)
    m,n = k.compute_confusion_matrix()


