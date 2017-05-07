import pandas as pd
import numpy as np
import logistic_classifier as lc
import compute_confusion_matrix2 as cc


X_raw_test = np.load('C:/Users/Chaomin/Desktop/data_mining/data/result/X_raw_test.npy')
y_test = np.load('C:/Users/Chaomin/Desktop/data_mining/data/result/y_test.npy')
V_t_1000 = np.load('C:/Users/Chaomin/Desktop/data_mining/data/result/v_t_1000.npy')
V_1000 = np.load('C:/Users/Chaomin/Desktop/data_mining/data/result/V_1000.npy')
label = np.unique(y_test)

X_test = np.dot(X_raw_test,V_1000.T)

for i in range(10):
    logit_cl  = lc.LogisticClassifier(label)
    weights_address = 'C:/Users/Chaomin/Desktop/data_mining/data/intermediate/result/win/weight' + str(i) + '.npy'
    weights = np.load(weights_address)
    prob, in_matrix = logit_cl.predict_prob(X_test, weights, multiclass=True)
    result_list = logit_cl.predict_label(prob, 0.5, multiclass=True)
    predict_result_array = np.array(result_list, dtype=str)

    matrix_withheader_address = 'C:/Users/Chaomin/Desktop/data_mining/data/intermediate/result/wint/confusion_matrix_withheader' + str(i) + '.csv'
    matrix_address = 'C:/Users/Chaomin/Desktop/data_mining/data/intermediate/result/wint/confusion_matrix' + str(i) + '.csv'
    acc_matrix_address = 'C:/Users/Chaomin/Desktop/data_mining/data/intermediate/result/wint/acc' + str(i) + '.csv'
    confusion_matrix_builder = cc.confusion_matrix_builder(y_test, predict_result_array, label)
    confusion_matrix, label_list = confusion_matrix_builder.compute_confusion_matrix()
    confusion_matrix_builder.save_file(confusion_matrix, label_list, matrix_withheader_address, matrix_address)
    acc_matrix = cc.calculate_r_c_sum(confusion_matrix)

    row_header_l = list(['binary_classifier_by_label', 'TP', 'FN', 'FP', 'TN', 'ACC', 'SPE', 'PRE', 'F1'])
    column_extra_string = 'binary_classifier_by_label'
    result_df = cc.calculate_acc(confusion_matrix, acc_matrix, row_header_l, column_extra_string, label_list, data_sample_size=len(y_test))
    # result_df.to_csv('C:/Users/Chaomin/Desktop/data_mining/data/intermediate/result/acc.csv', index=False)

    result_df.to_csv(acc_matrix_address, index=False)
    print('training model {0} finishes !'.format(i + 1))