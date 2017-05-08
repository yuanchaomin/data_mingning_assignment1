import numpy as np
import pandas as pd
from sklearn import preprocessing
import time

class DataLoader:
    def __init__(self,data_address, label_address, predict_matrix_address):
        """
        
        :param data_address: the file address for training data(without label)
        :param label_address: the file address for training data(label)
        :param predict_matrix_address: the file address for testing data(without label)
        
        the output of this initialization:
        self.data : a ndarray with shape(m,n) where m is sample size and n is feature size
        self.label : a ndarray with shape(m,) where m is sample size
        self.predict_data: a ndarray with shape(m_1, n) where m_1 is sample size in test data, and n is feature size
        self.unique_label: a ndarray with shape(k,) where k is the size of unique label
        """
        self.data = np.genfromtxt(data_address, delimiter=',')
        self.label = np.genfromtxt(label_address, delimiter=',', dtype = str)
        self.predict_data = np.genfromtxt(predict_matrix_address, delimiter=',')
        self.unique_label = np.unique(self.label)
    def scale(self):
        self.data = preprocessing.scale(self.data)
        self.predict_data = preprocessing.scale(self.predict_data)

    def return_value(self):
        return self.data, self.label, self.predict_data, self.unique_label


class LogisticClassifier:
    def __init__(self, label):
        self.author = 'Chaomin'
        self.unique_label = label
        self.weights = None

    def generate_single_class_label(self, y_train_list, label_class):
        """
        This function will generate a list of labels as y_train for a binary classifier, which will be involved in 
        'one vs all' method later. The label_class specifies the 'one' class 
        :param label_list: the list of all label, that is, a list of y_train
        :param label_class: the 'one' class
        :return: a list of labels as y_train for a binary classifier
        """
        unique_label = list(np.unique(y_train_list))
        label_list_index = self.get_label_index(y_train_list,unique_label)

        return_list = []
        for i in label_list_index:
            if i == int(label_class):
                return_list.append(1)
            else:
                return_list.append(0)

        return return_list

    @staticmethod
    def get_label_index(label_list, label_unique_list):
        """
        
        :param label_list: a list of labels
        :param label_unique_list: a set of labels, stored in a list
        :return: the index of label_unique_list, which appears in label_list
        """

        a_list = []
        for i in label_list:
            a_list.append(label_unique_list.index(i))
        return a_list

    @staticmethod
    def sigmoid(x):
        return 1.0 / (1 + np.exp(-x))

    def fit(self,X_train, y_train, multiclass = False):
        """X_train : a matrix with shape(m,n), 
           y_train: target value, in binary situation, its value is 0 or 1 """
        if multiclass == True:
            y_unique_label = list(np.unique(y_train))
            y_unique_label_index = self.get_label_index(y_unique_label, y_unique_label)
            m,n = np.shape(X_train)
            j = len(y_unique_label)

            label_matrix = np.matrix(np.zeros((m,j)))
            weight_matrix = np.matrix(np.zeros((n,j)))
            #o = self.generate_single_class_label(list(y_train), 0)
            """ label_matrix: each row stands for a sample, each column stand for a class_label, for example , the first
            column means that this function will treat all samples with label 'class_one' as 1 and 0 otherwise.
            
            weight_matrix: each column stands for weight vectors for each class, for example, the first column means
            that class_one binary classifier has such weight vector. The size of row represents the number of features.
            
            
            """

            for i in y_unique_label_index:
                label_matrix[:,i] = np.matrix(self.generate_single_class_label(list(y_train), i)).T
            #k = self.gradient_ascent(X_train, label_matrix[:,0])
            #o = []
            #for i in range(3):
                #o.append(self.gradient_ascent(X_train, label_matrix[:,i]))
            for i in y_unique_label_index:
                weight_matrix[:,i] = self.gradient_ascent(X_train, label_matrix[:,i])

            self.weights = weight_matrix
            self.unique_label = y_unique_label
            self.unique_label_index = y_unique_label_index
            return weight_matrix

    def gradient_ascent(self,X_train,y_train):
        """
        This function will perform an gradient_ascent method , and return the weight vector
        :param X_train: Training data with shape(m,n), where m is sample size and n is feature size
        :param y_train: Target value, neither int 1 or 0
        :return: a weight vector with shape(1, n), where n is feature size
        """

        m,n = np.shape(X_train)
        alpha = 0.1
        max_steps = 100
        weights = np.ones((n,1))
        for i in range(max_steps):
            y_bar = self.sigmoid(np.dot(X_train, weights))
            error = y_train - y_bar
            weights = weights + alpha * X_train.T * error
        return weights

    def stochastic_gradient_ascent(self,max_steps):
        matrix = self.data
        label_value = self.label.T
        m, n = np.shape(matrix)
        weights = np.ones((1,n))
        for j in range(max_steps):
            dataIndex = range(m)
            for i in range(m):
                alpha = 4/(1.0 + i + j) + 0.01
                randIndex = int(np.random.uniform(0, len(dataIndex)))
                y_bar = self.sigmoid(np.sum(np.dot(matrix[randIndex], weights.T)))
                error = label_value[randIndex] - y_bar
                weights = weights + alpha * error * matrix[randIndex]
                np.delete(dataIndex, randIndex)
        return weights.T

    def predict_prob(self, X_test, weights, multiclass = False):
        if multiclass:
            inter_matrix = np.dot(X_test,weights)
            prob = self.sigmoid(inter_matrix)
            return prob, inter_matrix

        else:
           prob = self.sigmoid(X_test * weights)


        return prob

    def predict_label(self, prob, threshold, multiclass=False):
        if multiclass:
            m,n = np.shape(prob)
            # j = len(y_train)
            result_array = np.ones((m,1))
            result_list_final = []
            y_unique_label_list = list(self.unique_label)

            for i in range(m):
                result_array[i] = np.argmax(prob[i,:])

            result_list = result_array.tolist()

            for i in result_list:
                for j in i:
                    result_list_final.append(y_unique_label_list[int(j)])

            return result_list_final
        else:
            prob_to_label_f = np.vectorize(lambda x : 1 if x >= float(threshold) else 0)
            predict_label =  prob_to_label_f(prob)
            return predict_label

# if __name__  == "__main__":
#     #data_address = 'C:/Users/Chaomin/Desktop/data_mining/data/classifier_data_for_test/test_data_for_logit_r/logit_train_data.csv'
#     data_address = 'C:/Users/Chaomin/Desktop/data_mining/data/intermediate/try_X_train.csv'
#     label_address = 'C:/Users/Chaomin/Desktop/data_mining/data/classifier_data_for_test/test_data_for_logit_r/logit_train_labelCopy.csv'
#     #predict_matrix_address = 'C:/Users/Chaomin/Desktop/data_mining/data/classifier_data_for_test/test_data_for_logit_r/logit_test_data.csv'
#     predict_matrix_address = 'C:/Users/Chaomin/Desktop/data_mining/data/intermediate/try_X_train.csv'
#     data_loader = DataLoader(data_address, label_address, predict_matrix_address)
#     #data_loader.scale()
#     data, label, predict_matrix, unique_label = data_loader.return_value()
#
#     # np.delete(data, 0, axis = 0)
#     # np.delete(label,0, axis = 0)
#     X_train = data[1:,1:]
#     X_test = predict_matrix[1:,1:]
#
#     logit_cl = LogisticClassifier()
#
#     start = time.clock()
#     # X_train = data
#     #y_train = list(label)
#     # y_train = label
#     #y_train = np.matrix(label, dtype=float).T
#     #weights = logit_cl.gradient_ascent(X_train, y_train)
#     #weights= logit_cl.stochastic_gradient_ascent(1000)
#     #print(weights)
#     #X_train = np.matrix(data)
#     #y_train = np.matrix(label)
#     #a =  logit_cl.generate_single_class_label(y_train, 1)
#
#     # X_test = predict_matrix
#     # weights = logit_cl.fit(X_train, y_train, multiclass=True)
#     # prob,in_matrix = logit_cl.predict_prob(X_test,weights,multiclass=True)
#     # result_list = logit_cl.predict_label(prob,0.5, multiclass=True)
#
#     #print(np.shape(matrix_rand))
#     #result_prob = logit_cl.predict_prob(predict_matrix, weights)
#     #predict_label = logit_cl.predict_label(predict_matrix,weights,'0.5')
#     #print(weights)
#     #print(result_prob)
#     #print(predict_label)
#     end = time.clock()
#     print('Running time: %s Seconds' % (end - start))

if __name__  == "__main__":
    s_1000 = np.load('C:/Users/Chaomin/Desktop/data_mining/data/intermediate/my_s_1000_np_float64.npy')
    U_1000 = np.load('C:/Users/Chaomin/Desktop/data_mining/data/intermediate/my_U_1000_np_float64.npy')
    V_1000 = np.load('C:/Users/Chaomin/Desktop/data_mining/data/intermediate/my_V_1000_np_float64.npy')

    X_train_df = pd.read_csv('C:/Users/Chaomin/Desktop/data_mining/data/intermediate/X_train.csv')
    X_test_df = pd.read_csv('C:/Users/Chaomin/Desktop/data_mining/data/intermediate/X_test.csv')

    X_train_mat = X_train_df.as_matrix()
    X_test_mat = X_test_df.as_matrix()

    m_train, n_train = np.shape(X_train_mat)
    m_test, n_test = np.shape(X_test_mat)

    X_train_label = X_train_mat[:, n_train - 1]
    X_train_label = X_train_label.astype(str)
    X_test_label = X_test_mat[:, n_test - 1]
    X_test_label = X_test_label.astype(str)

    X_raw_train = X_train_mat[:, 1: n_train - 1]
    X_raw_train = X_raw_train.astype(float)
    X_raw_test = X_test_mat[:, 1: n_train - 1]
    X_raw_test = X_raw_test.astype(float)

    X_train = np.dot(X_raw_train, V_1000.T)
    X_test = np.dot(X_raw_test, V_1000.T)
    y_train = X_train_label
    y_test = X_test_label

