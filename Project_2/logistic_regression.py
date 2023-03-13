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
lambda(lamb) - the penalty term used in regularization
delta - a k x m matrix where delta_ji = 1 if jth training value, Y^i = y_j and 
        delta_ji = 0 otherwise
X - an m x (n+1) training set without index or class columns, 1 based index,
    m is rows in training set
Y - an m x 1 vector(matrix) of true classifications for each example
W - a k x (n+1) matrix of weights
"""
iter = 1000
eta = 0.1
lamb = 0.01
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

X_class = sparse.csr_matrix(train_sparse)

#number of classes or target values
k = np.max(train_sparse[:, -1])

#number of attibutes or features
n = X.shape[1]

#initialize weight matrix with zeros
W = np.zeros((k, n), dtype=float)

#making delta array with Y, all 0 except indecies with value (1, 2, or 3)
delta = np.zeros((k, X.shape[0]), dtype=int)
for i in range(16):
    if Y[i] == 1:
        delta[0][i] = 1
    elif Y[i] == 2:
        delta[1][i] = 1
    else:
        delta[2][i] = 1
print(delta)

#making prediction array
def make_pred(W):
    prediction = np.exp(np.dot(W,np.transpose(X)))
    prediction[k-1, :] = 1
    pt = np.transpose(prediction)
    prediction = np.transpose(pt/pt.sum(axis=1)[:, None])
    return prediction

#initialize likelihood sum
lh_sum = 0

def likelihood(W, X, Y):
    lh_sum = 0
    for i in range(X.shape[0]):
        a = int((Y[i]))
        # print(a)
        lh = a * (np.dot(W[a-1], X[i])) - \
            np.log(1+np.exp(np.dot(W[a-1], X[i])))
        lh_sum += lh
    return lh_sum


lh_sum = likelihood(W, X, Y)
current_weights = W.copy()
for i in range(iter):
    W = W + eta * (np.dot((delta - make_pred(W)), X) - (lamb * W))
    lh_new = likelihood(W, X, Y)
    if lh_sum < lh_new:
        lh_sum = lh_new
        current_weights = W.copy()
    print(lh_sum)


a = make_pred(current_weights)



for i in range(16):
    print("%.4f" % Y[i], end=" ")
print()

for i in range(16):
    print("%.4f" % a[0][i], end=" ")
print()
for i in range(16):
    print("%.4f" % a[1][i], end=" ")
print()
for i in range(16):
    print("%.4f" % a[2][i], end=" ")
