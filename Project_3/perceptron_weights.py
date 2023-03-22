import numpy as np

def xiW_T(xi, W):
    xiWT = np.dot(X, np.transpose(W))
    print(xiWT)
    return xiWT

def delta_W(y, y_pred, eta, X):
    deltaW = X.copy
    for i in range(len(X)):
        deltaW[i] = (y - y_pred) * eta * X[i]
    print(deltaW)
    return deltaW

eta = 1
W = np.array([0, 0, 0, 0])
X = np.array(['00110', '11110', '10111', '01110', '11011', '00011'])

for i in range(X.shape[0]):
    if xiW_T(X[i], W) > 0:
        y_pred = 1
    else:
        y_pred = 0
    print(y_pred)
    y = X[i, -1]
    W += delta_W(y, y_pred, eta, X[i])
