import numpy as np
from numpy import linalg as LA

def eigend(original_matrix):
    original_matrix_mean = original_matrix.mean(axis = 0)
    corelation_mat = np.cov((original_matrix - original_matrix_mean).T)
    eigen_values, eigen_matrix = LA.eig(corelation_mat)

    # idx = eigen_values.argsort()[::-1]
    # eigen_values = eigen_values[idx]
    # eigen_matrix = eigen_matrix[:, idx]

    return eigen_values, eigen_matrix.T, corelation_mat

def filter_eigenv(dictionary, begin, end):
  return {k: v for k, v in sorted(dictionary.items(), reverse=True) if (k >= begin) & (k <= end)}