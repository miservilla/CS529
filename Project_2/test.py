import pandas as pd
from scipy.sparse import csr_matrix
from scipy import sparse
import numpy as np
from numpy import genfromtxt
from numpy import asarray
from numpy import savetxt
from numpy import loadtxt
import math

lh_np = loadtxt('lh_np.csv', delimiter=',')
prior = lh_np[:, 61188]
lh_np = np.delete(lh_np, 61188, 1)
test_sparse = sparse.load_npz('Project_2/csr_test.csv.npz')
test = sparse.csr_matrix(test_sparse)
test_np = csr_matrix.toarray(test)
test_np = np.delete(test_np, 0, 1)  # delete 1st column (index)

product = [[0 for x in range(20)] for y in range(6774)]
product = np.array(product)

result = []

for i in range(6774):
    a = test_np[i]
    b = (a * lh_np)
    e = []
    for j in range(20):
        x = b[j]
        x = x[x != 0]
        x = np.log(x)
        c = np.prod(x, axis=0)
        d = (prior[j] * c)   
        e.append(d) 
    result.append(np.argmax(e))
    del e

print(result)
        
