import numpy as np
import pandas as pd
from sklearn import preprocessing

class DataLoader:
    def __init__(self,data_address, label_address, predict_matrix_address):
        self.data = np.genfromtxt(data_address, delimiter=',')
        self.label = np.genfromtxt(label_address, delimiter =',')
        self.predict_matrtix = np.genfromtxt(predict_matrix_address, delimiter = ',')
    def return_value(self):
        return self.data, self.label, self.predict_matrtix


class LogisticClassifier:
    def __init__(self, data, label):
        self.data = preprocessing.scale(np.matrix(data))
        self.label = np.matrix(label)

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1 + np.exp(-x))


    def gradient_ascent(self):
        matrix = self.data
        label_value = self.label.T
        m,n = np.shape(matrix)
        alpha = 0.0001
        max_steps = 100000
        weights = np.ones((n,1))
        for i in range((max_steps)):
            y_bar = self.sigmoid(np.dot(matrix, weights))
            error = label_value - y_bar
            weights = weights + alpha * matrix.T * error
        return weights

    def stochastic_gradient_ascent(self,max_steps):
        matrix = self.data
        label_value = self.label.T
        m, n = np.shape(matrix)
        weights = np.ones((1,n))
        for j in range(max_steps):
            dataIndex = range(m)
            for i in range(m):
                alpha = 4/(1.0 + i + j) + 0.001
                randIndex = int(np.random.uniform(0, len(dataIndex)))
                y_bar = self.sigmoid(np.sum(np.dot(matrix[randIndex], weights.T)))
                error = label_value[randIndex] - y_bar
                weights = weights + alpha * error * matrix[randIndex]
                np.delete(dataIndex, randIndex)
        return weights.T



    def predict_prob(self, predict_matrix, weights):
        self.prob = self.sigmoid(preprocessing.scale(predict_matrix) * weights)

        return self.prob

    def predict_label(self, predict_matrix, weights, threshold):
        self.prob = self.sigmoid(preprocessing.scale(predict_matrix) * weights)
        prob_to_label_f = np.vectorize(lambda x : 1 if x >= float(threshold) else 0)
        self.predict_label =  prob_to_label_f(self.prob)

        return self.predict_label

if __name__  == "__main__":
    data_address = 'C:/Users/Chaomin/Desktop/data_mining/data/classifier_data_for_test/test_data_for_logit_r/logit_train_data.csv'
    label_address = 'C:/Users/Chaomin/Desktop/data_mining/data/classifier_data_for_test/test_data_for_logit_r/logit_train_label.csv'
    predict_matrix_address = 'C:/Users/Chaomin/Desktop/data_mining/data/classifier_data_for_test/test_data_for_logit_r/logit_test_data.csv'
    data_loader = DataLoader(data_address, label_address, predict_matrix_address)

    data, label, predict_matrix = data_loader.return_value()

    # np.delete(data, 0, axis = 0)
    # np.delete(label,0, axis = 0)

    logit_cl = LogisticClassifier(data, label)

    #weights = logit_cl.gradient_ascent()
    weights= logit_cl.stochastic_gradient_ascent(150)
    print(weights)
    #print(np.shape(matrix_rand))
    #result_prob = logit_cl.predict_prob(predict_matrix, weights)
    #predict_label = logit_cl.predict_label(predict_matrix,weights,'0.5')
    #print(weights)
    #print(result_prob)
    #print(predict_label)

