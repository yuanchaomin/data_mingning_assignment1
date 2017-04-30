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
 

true_label_file_address = 'C:/Users/Chaomin/Desktop/data_mining/assignment1_2017S1//train_and_test_data/my_test_label.csv'
predict_label_file_address = 'C:/Users/Chaomin/Desktop/data_mining/assignment1_2017S1/result/logit_test_label.csv'
label_address = 'C:/Users/Chaomin/Desktop/data_mining/assignment1_2017S1/group_label.csv'

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
    
    def compute_confusion_matrix(self,save,*args):
        if len(args) == 2:
            matrix_without_header = args[0]
            matrix_name = args[1]
        true_label = np.genfromtxt(self.true_label_file_address, delimiter = ',',dtype=str)
        predicted_label = np.genfromtxt(self.predicted_label_file_address, delimiter = ',', dtype= str)
        label_list = list(np.genfromtxt(label_address, delimiter = ',',dtype= str))
        confusion_matrix_ = confusion_matrix(true_label, predicted_label, labels = label_list)
        if save:
            np.savetxt(matrix_without_header, confusion_matrix_,fmt='%5d', delimiter=',')
            df_matrix = pd.DataFrame(confusion_matrix_)
            df_matrix.columns = label_list
            df_matrix.insert(0,'true_label||predicted_label', pd.Series(label_list))
            df_matrix.to_csv(matrix_name, index = False)
 
            
        return confusion_matrix_, label_list
    

if __name__ == '__main__':
    k = confusion_matrix_builder(true_label_file_address,predict_label_file_address,label_address)
    #m,n = k.compute_confusion_matrix(False)
    m,n = k.compute_confusion_matrix(True, 'old.csv','new.csv')


# I just need this to test my code
