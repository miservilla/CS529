import pandas as pd
from scipy.sparse import csr_matrix
from scipy import sparse
import numpy as np
from numpy import genfromtxt
from numpy import asarray
from numpy import savetxt


# df = pd.read_csv('/home/michaelservilla/CS529/Project_2/training.csv', sep=',')
# df = df.drop(df.columns[0], axis=1)

train = genfromtxt('/home/michaelservilla/CS529/Project_2/training.csv', delimiter=',')
test = genfromtxt(
    '/home/michaelservilla/CS529/Project_2/testing.csv', delimiter=',')
# train = np.delete(train,0,1)
print(train)
print(train[:, 0])
print(train.data.shape)

print(test)
print(test[:, 0])
print(test.data.shape)
# x = StandardScaler().fit_transform(train.data)
# print(x)
train_sparse = csr_matrix(train, dtype=np.int16)
test_sparse = csr_matrix(test, dtype=np.int16)

print("Training sparse matrix:")
print(train_sparse)
print(train_sparse.shape)

print("Testing sparse matrix:")
print(test_sparse)
print(test_sparse.shape)


sparse.save_npz('/home/michaelservilla/CS529/Project_2/csr_train.csv.npz', train_sparse)
sparse.save_npz(
    '/home/michaelservilla/CS529/Project_2/csr_test.csv.npz', test_sparse)
# train_sparse = sparse.load_npz('/home/michaelservilla/CS529/Project_2/csr_train.csv.npz')
