'''
training.csv shape(12000, 61190), column 1 is index, column 61190 is newsgroup
(1-20), columns 2-61188 are words in row vector. Note: Using index base = 1, not
 0. For consistency, consider change all to 0 based indexing.

 vocabulary.txt shape (61188, 1), one column with 61188 words. Index will map to 
 column number in training.csv. Note: Using index base = 1, not 0. For 
 consistency, consider change all to 0 based indexing.

newsgrouplabels.txt shape (20, 1), one column with 20 rows of newsgroup labels. 
Index will map to last column in training.csv (each word vector row in
training.csv will have a single news group attached). Note: Using index base = 
1, not 0. For consistency, consider change all to 0 based indexing.

testing.csv shape(not calculated), similar to training.csv except last column
with news group labels dropped.



'''


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
# print(train)

print(train.shape)
# print(train.data)
# print(train.indices[1])

# for i in range(120):
#     print(train[i, 61188])
# for i in range(120):
#     print(train.data[i])
print(train[11999, 61188])