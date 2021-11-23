import numpy as np
import csv
from scipy.sparse import csr_matrix
from scipy.io import loadmat
import os
import errno


def matlab_matrix(path, **kwargs):
    return loadmat(path, **kwargs)


def regular_matrix(path, **kwargs):
    return np.loadtxt(path, **kwargs)


def numpy_matrix(path, **kwargs):
    return np.load(path, **kwargs)


def sparse_matrix(path, column_count, delimiter=' ', index_value_delimiter=':'):
    """
    the column indices for row i are stored in indices[indptr[i]:indptr[i+1]]
    and their corresponding values are stored in data[indptr[i]:indptr[i+1]].
    return the numpy array (D,N)
    """
    indptr = [0]
    indices = []  # column indices
    data = []
    shape = [0, column_count]

    with open(path) as csvFile:
        reader = csv.reader(csvFile, delimiter=delimiter)

        for row in reader:
            indptr.append(indptr[-1] + len(row))
            shape[0] += 1
            for i in range(len(row)):
                try:
                    index, value = row[i].split(index_value_delimiter)
                    indices.append(int(index))
                    data.append(float(value))
                except ValueError:
                    indptr[-1] -= 1

    return csr_matrix((np.array(data), indices, indptr), shape).T.toarray()  # D,N


def sparse_binary_matrix(path, column_count, delimiter=' '):
    """
    :return:  return the numpy array (D,N) consist of 1 and 0
    """
    indptr = [0]
    indices = []
    shape = [0, column_count]

    with open(path) as csvFile:
        reader = csv.reader(csvFile, delimiter=delimiter)

        for row in reader:
            indptr.append(indptr[-1] + len(row))
            shape[0] += 1
            for i in range(len(row)):
                if row[i]:
                    indices.append(int(row[i]))
                else:
                    indptr[-1] -= 1

    return csr_matrix((np.ones(len(indices)), indices, indptr), shape).T.toarray()  # (D,N)


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as exception:
        if exception.errno != errno.EEXIST:  # the directory already exits
            raise
