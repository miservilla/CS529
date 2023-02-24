import pandas as pd
from scipy.sparse import csr_matrix
from scipy import sparse
import numpy as np
from numpy import genfromtxt
from numpy import asarray
from numpy import savetxt

train_sparse = sparse.load_npz('/home/sin_nombre/CS529/Project_2/csr_train.csv.npz')

train = sparse.csr_matrix(train_sparse)

print(train)
print(train.data)