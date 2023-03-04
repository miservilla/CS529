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

yk = newsgroup
yk_docs_cnt = number of docs in yk (count up all lines in each newsgroup)
total_docs = total number of docs (line count)
v_total = total length of vocabulary count
x_i = number of each individual word in yk (newsgroup)
yk_words = total number of words in yk (newsgroup)
lh_arr = likelihood (2d array with newsgroups as rows and words as columns, last
column is prior)

TODO



'''


import pandas as pd
from scipy.sparse import csr_matrix
from scipy import sparse
import numpy as np
from numpy import genfromtxt
from numpy import asarray
from numpy import savetxt
from numpy import loadtxt
import math

def mle(yk_docs_cnt, total_docs):
    return(yk_docs_cnt / total_docs) # prior

def alpha(v_total):
    return(1 + (1 / v_total)) # alpha

def map(x_i, yk_words, v_total):
    return ((x_i + (alpha(v_total) - 1)) / (yk_words + ((alpha(v_total)-1) * v_total))) # likelihood

lh = [[0 for x in range(61189)] for y in range(21)]

train_sparse = sparse.load_npz(
    'Project_2/csr_train.csv.npz')

train = sparse.csr_matrix(train_sparse)
total_docs = train.shape[0]

print(train.shape)
print(train.shape[0])
print(train.shape[1])

df = pd.DataFrame(columns=['index', 'newsgroups'])

for i in range(9600):
    df.loc[len(df.index)] = [i, train[i, 61189]] # df starts at index 0

dataframes = {} # Dictionary of dataframes with newsgroups indexes 
for i in range(1, 21):
    dataframes["df_{0}".format(i)] = df[df['newsgroups'] == i]

wrd_per_NG = csr_matrix((20,61189), dtype=int)

sparse_newsgroups = {} # Dictionary of sparse matrix for each newsgroup
v_total = 61188
count = 1

for dataframe in dataframes:
    print()
    print("News Group:" + str(dataframes[dataframe].iloc[0]['newsgroups']))
    print("Size: " + str(len(dataframes[dataframe])))
    yk_docs_cnt = len(dataframes[dataframe])
    index = str(dataframes[dataframe].iloc[0]['newsgroups'])

    # Summing each row associated with current news group
    sparse_newsgroups["{0}".format(index)] = csr_matrix((1,61190), dtype=int)
    for row in dataframes[dataframe]['index']:
        sparse_newsgroups[index] = train[row].toarray() + sparse_newsgroups[index]

    # Deleting 1st and last column as they are index and news groups (not words)
    sparse_newsgroups[index] = np.delete(sparse_newsgroups[index], 0)
    sparse_newsgroups[index] = np.delete(sparse_newsgroups[index], 61188)
    print(sparse_newsgroups[index][0,0:25])

    yk_words = np.sum(sparse_newsgroups[index]) # Sum of all words in current news group
    print(yk_words)

    for i in range(v_total):
        x_i = sparse_newsgroups[index][0, i]
        lh[count][i] = map(x_i, yk_words, v_total)

    lh[count][61188] = mle(yk_docs_cnt, total_docs)
    # print("Test MAP: " + str(lh[count][0]) + " with x_i of " + str(sparse_newsgroups[index][0, 0]))
    # print("word count of " + str(yk_words) + " and v_total of " + str(v_total))
    # print("MLE: " + str(lh[count][61188]))

 
    count += 1

print()
lh_np = np.asarray(lh)
lh_np = np.delete(lh_np, 0, 0) # delete row 0, all zero's
savetxt('Project_2/lh_np.csv', lh_np, delimiter=',')

test_sparse = sparse.load_npz('Project_2/csr_test.csv.npz')
test = sparse.csr_matrix(test_sparse)
test_np = csr_matrix.toarray(test)
test_np = np.delete(test_np, 0, 1) # delete 1st column (index)

product = [[0 for x in range(20)] for y in range(2400)]
product = np.array(product)

print(lh_np.shape)
print(test_np.shape)
print(product.shape)       
