import numpy as np


def ber(y_true, y_pred):
    # return accuracy of classification
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    labels, number = np.unique(y_true, return_counts=True)
    # mean of recall for every type pf label
    acc = [np.sum(np.logical_and([labels[i] == y_true], [labels[i] == y_pred]))/number[i] for i in range(len(labels))]
    return np.mean(acc)

