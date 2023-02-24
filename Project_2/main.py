import pandas as pd
from scipy.sparse import csr_matrix
from scipy import sparse
import numpy as np
from numpy import genfromtxt
from numpy import asarray
from numpy import savetxt

train_sparse = sparse.load_npz(
    '/home/michaelservilla/CS529/Project_2/csr_train.csv.npz')

train = sparse.csr_matrix(train_sparse)

print(train.shape)
print(train.data)
print(train.indices[1])
col_index = 0
for i in range(120):
    print(train[i,train.indices[col_index]])
    col_index += 1
