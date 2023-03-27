from scipy.sparse import csr_matrix
from scipy import sparse
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt
import numpy as np
from numpy import genfromtxt
from numpy import asarray
from numpy import savetxt
from numpy import loadtxt
import math
import time

train_sparse = sparse.load_npz(
    '/home/michaelservilla/CS529/Project_2/csr_train.csv_lg.npz')

X = sparse.csr_matrix(train_sparse[:, 1:-1])
X = X.toarray()

x_frequency = X.sum(0)
sum_words = x_frequency.sum() - x_frequency[0]
x_probability = x_frequency / sum_words
vocab = np.loadtxt('/home/michaelservilla/CS529/Project_2/vocabulary.txt', dtype=str)
for i in range(vocab.shape[0]):
    print(str(vocab[i]), str(x_probability[i]), str(x_probability.sum(0)))