from scipy.sparse import csr_matrix
from scipy import sparse
import numpy as np
from numpy import genfromtxt
from numpy import asarray
from numpy import savetxt
from numpy import loadtxt
import math
import time

"""
k - number of classes in training set
n - number of attributes (features or columns) each example (document or row)
    has
eta - the learning rate or step size
lambda_ - the penalty term used in regularization
delta - a k x m matrix where delta_ji = 1 if jth training value, Y^i = y_j and 
        delta_ji = 0 otherwise
X - an m x (n+1) training set without index or class columns, 1 based index,
    m is rows in training set
Y - an m x 1 vector(matrix) of true classifications for each example
W - a k x (n+1) matrix of weights
"""
start_time = time.time()
iter = 1
eta = 0.001
lambda_ = 0.001
#load compressed training set
train_sparse = sparse.load_npz(
    '/home/michaelservilla/CS529/Project_2/csr_train.csv_sm.npz')

# print(train_sparse.shape)
# print(train_sparse)

#load compressed testing set
test_sparse = sparse.load_npz(
    '/home/michaelservilla/CS529/Project_2/csr_test.csv_sm.npz')

print(test_sparse.shape)
print(test_sparse)


#vector of true classes for each example (row) test
Y_test = test_sparse[:, -1]
Y_test = Y_test.toarray()
print(Y_test.shape)
print(Y_test)

#vector of true classes for each example (row) train
Y = train_sparse[:, -1]
Y = Y.toarray()
# print(Y.shape)
# print(Y)


#testing set without class column
X_test = sparse.csr_matrix(test_sparse[:, 0:-1])
X_test = X_test.toarray()
for i in range(X_test.shape[0]):
    X_test[i][0] = 1
#normalizing X_test
X_sum = X_test.sum(0)
X_sum = np.where(X_sum == 0, 1, X_sum)
X_test = X_test/X_sum

print(X_test.shape)
print(X_test)

#training set without class columns
X = sparse.csr_matrix(train_sparse[:, 0:-1])
X = X.toarray()
# X = X/100
for i in range(X.shape[0]):
    X[i][0] = 1
#nomralizing X
X_sum = X.sum(0)
X_sum = np.where(X_sum == 0, 1, X_sum)
X = X/X_sum

#number of classes or target values
k = len(np.unique(Y))

#number of attibutes or features
n = X.shape[1]

#initialize weight matrix with zeros
W = np.zeros((k, n), dtype=float)

#delta array with Y, all 0 except indices with values of target (one hot encoding)
delta = np.zeros((k, X.shape[0]), dtype=int)
for i in range(Y.shape[0]):
    row = Y[i]
    delta[row[0]-1][i] = 1

#making prediction array
def make_pred(W, X_array):
    prediction = np.exp(np.dot(W,np.transpose(X_array)))
    prediction[k-1, :] = 1
    prediction = prediction/prediction.sum(0)
    return prediction

#makes conditional likelihood estimate
def likelihood(W, X, Y):
    lh_sum = 0
    for i in range(X.shape[0]):
        a = int((Y[i]))
        lh = a * (np.dot(W[a-1], X[i])) - np.log(1+np.exp(np.dot(W[a-1], X[i])))
        lh_sum += lh
    return lh_sum

#initialize likelihood sum
lh_sum = likelihood(W, X, Y)

#saves initial current weights
current_weights = W.copy()

#runs gradient ascent
count = 0
for i in range(iter):
    W = W + eta * (np.dot((delta - make_pred(W, X)), X) - (lambda_ * W))
    lh_new = likelihood(W, X, Y)
    # difference = abs(lh_new - lh_sum)
    # if difference < 0.00001:
    #     break
    if lh_sum < lh_new:
        lh_sum = lh_new
        current_weights = W.copy()
    count += 1
    # print(count, lh_sum, difference)
    print(count, lh_sum)

savetxt('logistic_regression_wts.csv', current_weights, delimiter=",")

a = make_pred(current_weights, X)

b = make_pred(current_weights, X_test)

accuracy = 0
for i in range(b.shape[1]):
    b_max = np.argmax(b, axis=0)
    if int(Y_test[i]) == (b_max[i] + 1):
        accuracy += 1

print("accuracy = ", str(accuracy/b.shape[1]))
print("--- %s seconds ---" % (time.time() - start_time))
