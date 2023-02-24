import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import csr_matrix
from sklearn import datasets
import numpy as np
from numpy import count_nonzero
from numpy import genfromtxt

# load the inbuild digits dataset
digits = datasets.load_digits()

print(type(digits))
print(digits.data)

# shape of the dense matrix
print(digits.data.shape)

# standardizing the data points
X = StandardScaler().fit_transform(digits.data)
print(X)

# representing in CSR form
X_sparse = csr_matrix(X)
print(X_sparse)

# specify the no of output features
tsvd = TruncatedSVD(n_components=10)

# apply the truncatedSVD function
X_sparse_tsvd = tsvd.fit(X_sparse).transform(X_sparse)
print(X_sparse_tsvd)

# shape of the reduced matrix
print(X_sparse_tsvd.shape)
