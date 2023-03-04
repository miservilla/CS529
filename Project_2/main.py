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
    ********* See todo lower in document for what to work on next, it's in the
              location that it should go inside the document                   **************



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


# Loading training data 
train_sparse = sparse.load_npz(
    'Project_2/csr_train.csv.npz')


# Adjust these global values for specific training set
train = sparse.csr_matrix(train_sparse) # Traing data as sparce matrix
unique_targets = 20 # Number of possible targets (number of news groups)
total_docs = train.shape[0] # Number of documents in trianing set
v_total = 61188 # Number of unique words in training set
column_count = train.shape[1] # Number of columns prior to removing unneeded index and target columns


def mle(yk_docs_cnt, total_docs):
    return(yk_docs_cnt / total_docs) # prior

def alpha(v_total):
    return(1 + (1 / v_total)) # alpha

def map(x_i, yk_words, v_total):
    return ((x_i + (alpha(v_total) - 1)) / (yk_words + ((alpha(v_total)-1) * v_total))) # likelihood

def build_MAP_MLE(lh, beta): #Uses global values
    df = pd.DataFrame(columns=['index', 'newsgroups'])

    for i in range(total_docs):
        df.loc[len(df.index)] = [i, train[i, 61189]] # df starts at index 0

    dataframes = {} # Dictionary of dataframes with newsgroups indexes 
    for i in range(1, unique_targets+1):
        dataframes["df_{0}".format(i)] = df[df['newsgroups'] == i]

    wrd_per_NG = csr_matrix((unique_targets,v_total+1), dtype=int)

    sparse_newsgroups = {} # Dictionary of sparse matrix for each newsgroup
    count = 1

    for dataframe in dataframes:
        print()
        print("News Group:" + str(dataframes[dataframe].iloc[0]['newsgroups']))
        print("Size: " + str(len(dataframes[dataframe])))
        yk_docs_cnt = len(dataframes[dataframe])
        index = str(dataframes[dataframe].iloc[0]['newsgroups'])

        # Summing each row associated with current news group
        sparse_newsgroups["{0}".format(index)] = csr_matrix((1,column_count), dtype=int)
        for row in dataframes[dataframe]['index']:
            sparse_newsgroups[index] = train[row].toarray() + sparse_newsgroups[index]

        # Deleting 1st and last column as they are index and news groups (not words)
        sparse_newsgroups[index] = np.delete(sparse_newsgroups[index], 0)
        sparse_newsgroups[index] = np.delete(sparse_newsgroups[index], v_total)
        print(sparse_newsgroups[index])
        
        yk_words = np.sum(sparse_newsgroups[index]) # Sum of all words in current news group
        
        for i in range(v_total):
            x_i = sparse_newsgroups[index][0, i]
            lh[count][i] = map(x_i, yk_words, beta)

        lh[count][61188] = mle(yk_docs_cnt, total_docs)
        # print("Test MAP: " + str(lh[count][0]) + " with x_i of " + str(sparse_newsgroups[index][0, 0]))
        # print("word count of " + str(yk_words) + " and v_total of " + str(v_total))
        # print("MLE: " + str(lh[count][61188]))

    
        count += 1
    return lh


# Getting MAP and MLE values for beta = v_total 
lh = build_MAP_MLE([[0 for x in range(v_total+1)] for y in range(unique_targets+1)], beta=v_total) 

# Turning lh into an array and then saving it to a file (np ~ numpy)
lh_np = np.asarray(lh)
lh_np = np.delete(lh_np, 0, 0) # delete row 0, all zero's
savetxt('Project_2/lh_np.csv', lh_np, delimiter=',')


'''
TODO here 

    Need to create a loop to make and *save MAP_MLE array for beta values 0.00001 to 1
    not sure if we actually want to be saving that many files or convert code in test
    file to a function that can be called in main....See question 2 in the PDF for Project
    2 for more clarification on what I am talking about


'''
