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
lambda_ - the penalty term used in regularization
delta - a k x m matrix where delta_ji = 1 if jth training value, Y^i = y_j and 
        delta_ji = 0 otherwise
X - an m x (n+1) training set without index or class columns, 1 based index,
    m is rows in training set
Y - an m x 1 vector(matrix) of true classifications for each example
W - a k x (n+1) matrix of weights
"""
iter = 1000
eta = 0.1
lambda_ = 0.01
#load compressed training set
train_sparse = sparse.load_npz(
    'Project_2/csr_train.csv_lg.npz')

# print(train_sparse.shape)
# print(train_sparse)

#load compressed testing set
test_sparse = sparse.load_npz(
    'Project_2/csr_test.csv_lg.npz')

# print(test_sparse.shape)
# print(test_sparse)


#vector of true classes for each example (row) test
Y_test = test_sparse[:, -1]
Y_test = Y_test.toarray()
print("Y_test created")
# print(Y_test.shape)
# print(Y_test)

#vector of true classes for each example (row) train
Y = train_sparse[:, -1]
Y = Y.toarray()
print("y created")
# print(Y.shape)
# print(Y)


#testing set without class column
X_test = sparse.csr_matrix(test_sparse[:, 0:-1])
X_test[:,0] = np.ones(X_test.shape[0])
X_test = X_test.toarray()
print("X_test created")


# print(X_test.shape)
# print(X_test)

#training set without class columns
X = sparse.csr_matrix(train_sparse[:, 0:-1])
X[:,0] = np.ones(X.shape[0])
X = X/1000
X = X.toarray()

# X = sparse.csr_matrix(sparse.load_npz(
#     'Project_2/csr_train.csv_lg_reduced.npz')).toarray()
# print("X created")

#number of classes or target values
k = len(np.unique(Y))

#number of attibutes or features
n = X.shape[1]

#initialize weight matrix with zeros
W = np.zeros((k, n), dtype=float)


#making delta array with Y, all 0 except indecies with value (1, 2, or 3)
delta = np.zeros((k, X.shape[0]), dtype=int)
for i in range(Y.shape[0]):
    row = Y[i]
    delta[row[0]-1][i] = 1


#making prediction array
def make_pred(W, X_array):
    prediction = np.exp(np.dot(W,np.transpose(X_array)))
    prediction[k-1, :] = 1
    pt = np.transpose(prediction)
    prediction = np.transpose(pt/pt.sum(axis=1)[:, None])
    return prediction


#makes conditional likelihood estimate
def likelihood(W, X, Y):
    lh_sum = 0
    for i in range(X.shape[0]):
        a = int((Y[i]))
        b = np.dot(W[a-1], X[i])
        c = np.log(1+np.exp(np.dot(W[a-1], X[i])))
        lh = a * b - c
        # lh = a * (np.dot(W[a-1], X[i])) - np.log(1+np.exp(np.dot(W[a-1], X[i])))
        lh_sum += lh
    return lh_sum


def calculate_weights(W, X, Y):

    #initialize likelihood sum
    lh_sum = likelihood(W, X, Y)

    #saves initial current weights
    current_weights = W.copy()

    #runs gradient ascent
    for i in range(iter):
        predict = make_pred(W, X)
        loss = delta - predict
        dot_loss = np.dot(loss, X)
        lambda_W = lambda_ * W
        dot_loss_lambda_W = dot_loss - lambda_W
        W = W + eta * (dot_loss - lambda_W)

        # W = W + eta * (np.dot((delta - make_pred(W, X)), X) - (lambda_ * W))
        lh_new = likelihood(W, X, Y)
        if lh_sum < lh_new:
            lh_sum = lh_new
            current_weights = W.copy()
        print(lh_sum)

    return current_weights

def print_prediction(prediction):
    print("Prediction: ------------------------------------------------")
    rows = prediction.shape[0]
    cols = prediction.shape[1]
    for i in range(rows):
        for j in range(cols):
            print(str(prediction[i][j])[:7], end=" ")
        print()

def class_prediction(prediction, target):
    predicted_class = np.zeros(target.shape[0], dtype=int)
    for i in range(prediction.shape[1]):
        tmp = prediction[:,i]
        location = np.unravel_index(np.argmax(tmp), tmp.shape)
        predicted_class[i] = location[0] + 1

    return predicted_class

def print_class_prediction(predicted_class, target):
    print()
    print("True target vs Predicted target: ----------------------------")
    print(np.transpose(target)[0])
    print(predicted_class)

def get_accuracy(predicted_class, target):
    length = len(predicted_class)
    correct = 0
    for i in range(length):
        if predicted_class[i] == target[i]:
            correct += 1
    return correct/length


weights = calculate_weights(W, X, Y)
prediction = make_pred(weights, X)
predicted_class = class_prediction(prediction, Y)
accuracy = get_accuracy(predicted_class, Y)

print_class_prediction(predicted_class, Y)
print(accuracy)

savetxt('weights.csv', weights, delimiter=',')