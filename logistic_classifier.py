import numpy as np
import pandas as pd


class DataLoader:
    def __init__(self,data_address, label_address, predict_matrix_address):
        self.data = np.genfromtxt(data_address, delimiter=',')
        self.label = np.genfromtxt(label_address, delimiter =',')
        self.predict_matrtix = np.genfromtxt(predict_matrix_address, delimiter = ',')
    def return_value(self):
        return self.data, self.label, self.predict_matrtix


class LogisticClassifier:
    def __init__(self, data, label):
        self.data = np.matrix(data)
        self.label = np.matrix(label)

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1 + np.exp(-x))

    def gradient_ascent(self):
        matrix = self.data
        label_value = self.label.T
        m,n = np.shape(matrix)
        alpha = 0.000001
        max_steps = 31000
        weights = np.ones((n,1))
        for i in range((max_steps)):
            y_bar = self.sigmoid(np.dot(matrix, weights))
            error = label_value - y_bar
            weights = weights + alpha * matrix.T * error
        return weights

    def predict_prob(self, predict_matrix, weights):
        self.prob = self.sigmoid(predict_matrix * weights)

        return self.prob

    def predict_label(self, predict_matrix, weights, threshold):
        self.predit_label = []
        self.prob = self.sigmoid(predict_matrix * weights)

        # for i in self.prob:
        #     if self.prob >= float(threshold):
        #
        #         return

if __name__  == "__main__":
    data_address = 'C:/Users/Chaomin/Desktop/data_mining/data/classifier_data_for_test/test_data_for_logit_r/logit_train_data.csv'
    label_address = 'C:/Users/Chaomin/Desktop/data_mining/data/classifier_data_for_test/test_data_for_logit_r/logit_train_label.csv'
    predict_matrix_address = 'C:/Users/Chaomin/Desktop/data_mining/data/classifier_data_for_test/test_data_for_logit_r/logit_test_data.csv'
    data_loader = DataLoader(data_address, label_address, predict_matrix_address)

    data, label, predict_matrix = data_loader.return_value()

    # np.delete(data, 0, axis = 0)
    # np.delete(label,0, axis = 0)

    logit_cl = LogisticClassifier(data, label)

    weights = logit_cl.gradient_ascent()

    result_prob = logit_cl.predict_prob(predict_matrix, weights)

    print(weights)
    print(result_prob)

