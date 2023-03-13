from scipy.sparse import csr_matrix
from scipy import sparse
import numpy as np
from numpy import genfromtxt
from numpy import asarray
from numpy import savetxt
from numpy import loadtxt
import math

"""
k - number of classes in training set
n - number of attributes (features or columns) each example (document or row)
    has
eta - the learning rate or step size
lambda - the penalty term used in regularization
delta - a k x m matrix where delta_ji = 1 if jth training value, Y^i = y_j and 
        delta_ji = 0 otherwise
X - an m x (n+1) training set without index or class columns, 1 based index,
    m is rows in training set
Y - an m x 1 vector(matrix) of true classifications for each example
W - a k x (n+1) matrix of weights
"""
iter = 10
eta = 0.001

#load compressed training set
train_sparse = sparse.load_npz(
    '/home/michaelservilla/CS529/Project_2/csr_train.csv_sm.npz')

# print(train_sparse.shape)
# print(train_sparse)

#vector of true classes for each example (row)
Y = train_sparse[:, -1]
Y = Y.toarray()

#training set without class columns
X = sparse.csr_matrix(train_sparse[:, 0:-1])
X = X.toarray()
for i in range(X.shape[0]):
    X[i][0] = 1



#number of classes or target values
k = np.max(train_sparse[:, -1])

#number of attibutes or features
n = X.shape[1]

#initialize weight matrix with zeros
W = np.zeros((k, n), dtype=float)

#training set without index or class columns to make delta array
delta = np.zeros((k, X.shape[0]), dtype=int)
X_less = sparse.csr_matrix(train_sparse[:, 1:-1])
print(X_less)
indices = X_less.tolil().rows
print(indices.shape)
print(indices)
print(indices[0][0])
print(len(indices[0]))
print(X_less[0,17])
# for i in range(indices.shape[0]):
#     for j in range(len(indices[i])):
#         print(i, j, indices[i][j])
#         print(X_less[i, indices[i][j]])
#         if X_less[i, indices[i][j]] == 1:
#             delta[0][indices[i][j]] = 1
#         elif X_less[i, indices[i][j]] == 2:
#             delta[1][indices[i][j]] = 1
#         else:
#             delta[2][indices[i][j]] = 1

print(delta)

#initialize likelihood sum
lh_sum = 0

# print(X.shape)
# print(X)
# print(Y)
# print(k)
# print(n)
# print(W.shape)
# print(X.shape)
# print(W[0])
# print()
# print(X[0])
# print(X.ndim)
# print(W.ndim)

def likelihood(W, X, Y):
    lh_sum = 0
    for i in range(X.shape[0]):
        a = int((Y[i]))
        # print(a)
        lh = a * (np.dot(W[a-1], X[i])) - \
            np.log(1+np.exp(np.dot(W[a-1], X[i])))
        lh_sum += lh
    print(lh_sum)

# for i in range(iter):
#     for j in range(W.shape[0]):
#         W[j] = W[j] + eta * ()
# likelihood(W, X, Y)