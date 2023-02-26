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




'''


import pandas as pd
from scipy.sparse import csr_matrix
from scipy import sparse
import numpy as np
from numpy import genfromtxt
from numpy import asarray
from numpy import savetxt

def mle(yk_docs_cnt, total_docs):
    return(yk_docs_cnt / total_docs)#prior

def alpha(v_total):
    return(1 + (1 / v_total))#alpha

def map(x_i, alpha, yk_words, v_total):
    return((x_i + (alpha - 1)) / (yk_words + (alpha * v_total)))#likelihood

lh = []


train_sparse = sparse.load_npz(
    '/home/michaelservilla/CS529/Project_2/csr_train.csv.npz')

train = sparse.csr_matrix(train_sparse)
df = pd.DataFrame(columns=['index', 'newsgroups'])

for i in range(12000):
    df.loc[len(df.index)] = [i, train[i, 61189]]

newsgroups_sorted = df.sort_values(by=['newsgroups'], ignore_index=True)
# print(newsgroups_sorted)
df_1 = df[df['newsgroups'] == 1]
# print(df_1)

dfs = {}
for i in range(1, 21):
    dfs["df_{0}".format(i)] = df[df['newsgroups'] == i]

print(dfs['df_13'])

print(train[0, 2])
# for i in range(1, 21):
#     for j in range(12000):
