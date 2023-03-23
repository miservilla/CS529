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
from numpy import loadtxt
import math


'''
Adjust these global variables for specific training set:
'''
# Loading training data 
train_sparse = sparse.load_npz(
    'Project_2/csr_train.csv_lg.npz')

train = sparse.csr_matrix(train_sparse) # Traing data as sparce matrix
# print(train.shape)

unique_targets = train[np.argmax(train[:, -1]), -1] # Number of possible targets
#(number of news groups) based on last column values

total_docs = train.shape[0] # Number of documents in trianing set

v_total = train.shape[1] - 2  # Number of unique words in training set, need to 
#drop index and class columns, so - 2

column_count = train.shape[1] # Number of columns prior to removing unneeded 
#index and target columns

run_loop = False # Set to false so that only a single beta value run is completed 
'''
End of global variables  
'''

'''
mle returns the prior for use in Naive Bayes
'''
def mle(yk_docs_cnt, total_docs):
    return(yk_docs_cnt / total_docs) # prior

'''
alpha returns the alpha value used by MAP 
beta is structured as 1/'beta' due to the original beta value being 1/|V| 
'''
def alpha(beta):
    return(1 + (1 / beta)) # alpha

'''
map returns the MAP estimate for use in Naive Bayes
'''
def map(x_i, yk_words, v_total, beta):
    return ((x_i + (alpha(beta) - 1)) / (yk_words + ((alpha(beta)-1) * v_total))) # likelihood

'''
build_MAP_MLE returns a matrix of MAP and MLE estimates for each class, the first 0 to v_total-1 columns
contain the MAP estimate for each attribute and the final column contains the MLE estimate
'''
def build_MAP_MLE(lh, beta): #Uses global values
    df = pd.DataFrame(columns=['index', 'class_id'])

    for i in range(total_docs):
        df.loc[len(df.index)] = [i, train[i, column_count - 1]] # df starts at index 0

    dataframes = {} # Dictionary of dataframes with newsgroups indexes 
    for i in range(1, unique_targets+1):
        dataframes["df_{0}".format(i)] = df[df['class_id'] == i]

    wrd_per_NG = csr_matrix((unique_targets,v_total+1), dtype=int)

    sparse_classes = {} # Dictionary of sparse matrix for each newsgroup
    count = 1

    for dataframe in dataframes:
        print()
        print("Class:" + str(dataframes[dataframe].iloc[0]['class_id'])) # Showing progress 
        # print("Size: " + str(len(dataframes[dataframe])))
        yk_docs_cnt = len(dataframes[dataframe])
        index = str(dataframes[dataframe].iloc[0]['class_id'])

        # Summing each row associated with current news group
        sparse_classes["{0}".format(index)] = csr_matrix((1,column_count), dtype=int)
        for row in dataframes[dataframe]['index']:
            sparse_classes[index] = train[row].toarray() + sparse_classes[index]

        # Deleting 1st and last column as they are index and news groups (not words)
        sparse_classes[index] = np.delete(sparse_classes[index], 0)
        sparse_classes[index] = np.delete(sparse_classes[index], v_total)
        # print(sparse_classes[index])
        
        yk_words = np.sum(sparse_classes[index]) # Sum of all words in current news group
        
        # Finding the MAP values for each attribute for the current class
        for i in range(v_total):
            x_i = sparse_classes[index][0, i]
            lh[count][i] = map(x_i, yk_words, v_total, beta)

        # Finding the MLE value for the current class
        lh[count][v_total] = mle(yk_docs_cnt, total_docs)
        # print("Test MAP: " + str(lh[count][0]) + " with x_i of " + str(sparse_newsgroups[index][0, 0]))
        # print("word count of " + str(yk_words) + " and v_total of " + str(v_total))
        # print("MLE: " + str(lh[count][61188]))

    
        count += 1
    return lh


'''
Body of main.py
'''
if not run_loop:

    # Getting MAP and MLE values for beta = v_total 
    lh = build_MAP_MLE([[0 for x in range(v_total+1)] for y in range(unique_targets+1)], beta=v_total) 

    # Turning lh into an array and then saving it to a file (np ~ numpy)
    lh_np = np.asarray(lh)
    lh_np = np.delete(lh_np, 0, 0) # delete row 0, all zero's
    savetxt('Project_2/lh_np.csv', lh_np, delimiter=',')

else:

    # Setting up a range for testing beta values
    beta_testing = {}
    beta_range = np.linspace(100000,1,50) # Beta values from 0.00001 to 1 
    #(Note: when used in the alpha function beta becomes 1/'assigned value')
    # print(beta_range)
    b_length = len(beta_range)

    # Getting MAP and MLE values for each beta value in beta_testing and saving them to a folder 
    for i in range(b_length):
        print(round(beta_range[i]))
        current_beta = build_MAP_MLE([[0 for x in range(v_total+1)] for y in range(unique_targets+1)], beta=round(beta_range[i])) 
        current_beta = np.asarray(current_beta)
        current_beta = np.delete(current_beta, 0, 0) # delete row 0, all zero's
        savetxt('Project_2/Diff_Beta_Values/beta_'+str(i)+'.csv', current_beta, delimiter=',')


