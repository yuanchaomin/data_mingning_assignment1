import numpy as np


class DataLoader:
    def __init__(self,data_address, label_address):
        self.data = np.genfromtxt(data_address, delimiter=',')
        self.label = np.genfromtxt(label_address, delimiter =',')

    def return_value(self):
        return self.data, self.label


class LogisticClassifier:
    def __init__(self, data, label):
        self.data = np.matrix(data)
        self.label = np.matrix(label)
        return

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1 + np.exp(-x))

    def gradient_ascent(self,data_matrix,label_matrix):
        matrix = data_matrix
        label_value = label_matrix.T
        m,n = np.shape(matrix)
        alpha = 0.001
        max_steps = 600
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