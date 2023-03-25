
import pandas as pd
from scipy.sparse import csr_matrix
from scipy import sparse
import numpy as np
from numpy import genfromtxt
from numpy import asarray
from numpy import savetxt
from numpy import loadtxt
import math




train_sparse = sparse.load_npz(
    'Project_2/csr_test.csv_sm.npz')

#training set without class columns
X = sparse.csr_matrix(train_sparse[:, 0:-1])
X = X.toarray()
# X = X/100
for i in range(X.shape[0]):
    X[i][0] = 1
#nomralizing X
X_sum = X.sum(0)
X_sum2 = np.where(X_sum == 0, 1, X_sum)
X = X/X_sum2

occurance = 1
maxValues = np.amax(X, axis=0)        
maxValues = np.where(X_sum < occurance, 0, maxValues) 

max_100 =  np.argpartition(maxValues,-15)[-15:]
max_100 = max_100[np.argsort(maxValues[max_100])]
max_100 = max_100[::-1]

print("X created")
print(max_100)

savetxt('Project_2/max_100_words.csv', max_100, delimiter=',', 
           fmt ='% s')