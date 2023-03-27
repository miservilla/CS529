from scipy.sparse import csr_matrix
from scipy import sparse
import numpy as np
from numpy import genfromtxt
from numpy import asarray
from numpy import savetxt
from sklearn.model_selection import train_test_split

# enter train csv file path
train = genfromtxt(
    'file path for training data in sparse csv format', delimiter=',')
# prediction = genfromtxt(
#     'file path for test (prediction) data in sparse csv format', 
# delimiter=',')

#comment out below if compressiong prediction file
split = round(len(train) * 0.80)
indices = np.random.permutation(train.shape[0])
training_idx, test_idx = indices[:split], indices[split:]
train, test = train[training_idx, :], train[test_idx, :]
train_sparse = csr_matrix(train, dtype=np.int16)
test_sparse = csr_matrix(test, dtype=np.int16)


# prediction_sparse = csr_matrix(prediction, dtype=np.int16)

# enter file path to save the train and test files
sparse.save_npz(
    'file path to save file/csr_train.csv.npz', train_sparse)
sparse.save_npz(
    'file path to save file/csr_test.csv.npz', test_sparse)

# sparse.save_npz(
#     'file path to save file/csr_prediction.csv.npz',
#     prediction_sparse)
