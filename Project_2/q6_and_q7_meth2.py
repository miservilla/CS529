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
    'csr_train.csv_sm.npz')

# Traing data as sparse matrix
X = sparse.csr_matrix(train_sparse)
print(X.shape)

# Number of possible targets (number of news groups) based on last column values
unique_targets = X[np.argmax(X[:, -1]), -1] 

# Number of documents in trianing set
total_docs = X.shape[0] 

# Number of unique words in training set, need to drop index and class columns, so - 2
v_total = X.shape[1] - 2  

# Number of columns prior to removing unneeded index and target columns
column_count = X.shape[1]

#training set without class columns
X = sparse.csr_matrix(train_sparse)
X = X.toarray()
for i in range(X.shape[0]):
    X[i][0] = 1
#normalizing X
X_sum = X.sum(0)
X_sum2 = np.where(X_sum == 0, 1, X_sum)
X = X/X_sum2
X[:,X.shape[1]-1] = np.transpose(train_sparse[:,X.shape[1]-1].toarray())
print(X)

df = pd.DataFrame(columns=['index', 'class_id'])

for i in range(total_docs):
    df.loc[len(df.index)] = [i, X[i, column_count - 1]] 
    
# Dictionary of dataframes with newsgroups indexes
dataframes = {}  
for i in range(1, unique_targets+1):
    dataframes["df_{0}".format(i)] = df[df['class_id'] == i]

sparse_classes = {} # Dictionary of sparse matrix for each newsgroup

for dataframe in dataframes:
    print()
    print("Class:" + str(dataframes[dataframe].iloc[0]['class_id']))
    print("Size: " + str(len(dataframes[dataframe])))
    # print(dataframes[dataframe])

    yk_docs_cnt = len(dataframes[dataframe])
    index = str(dataframes[dataframe].iloc[0]['class_id'])

    # Summing each row associated with current news group
    sparse_classes["{0}".format(index)] = csr_matrix((1,column_count), dtype=int)
    for row in dataframes[dataframe]['index']:
        sparse_classes[index] = X[int(row)] + sparse_classes[index]

    # Deleting 1st and last column as they are index and news groups (not words)
    sparse_classes[index] = np.delete(sparse_classes[index], 0)
    sparse_classes[index] = np.delete(sparse_classes[index], v_total)
    print(sparse_classes[index])

newsgroup_percents = np.zeros(shape=(unique_targets,v_total))
for i in range(1, unique_targets+1):
    newsgroup_percents[i-1,:] = sparse_classes[str(i)+".0"]

print()
print(newsgroup_percents)
print()

occurance = 1
maxValues = np.amax(newsgroup_percents, axis=0)  
X_sum_small = X_sum[0:-1]
X_sum_small = X_sum_small[1:]  
print(X_sum_small) 
maxValues = np.where(X_sum_small <= occurance, 0, maxValues) 

print(maxValues)

max_100 =  np.argpartition(maxValues,-15)[-15:]
max_100 = max_100[np.argsort(maxValues[max_100])]
max_100 = max_100[::-1]

print("X created")
print(max_100)

savetxt('max_100_words_meth2.csv', max_100, delimiter=',', 
           fmt ='% s')