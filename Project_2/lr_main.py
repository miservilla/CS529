import pandas as pd
from scipy.sparse import csr_matrix
from scipy import sparse
import numpy as np

# Loading training data 
train_sparse = sparse.load_npz(
    'Project_2/csr_train.csv.npz')


# Adjust these global values for specific training set
train = sparse.csr_matrix(train_sparse) # Traing data as sparce matrix
print(train.shape)

unique_targets = train[np.argmax(train[:, -1]), -1] # Number of possible targets
#(number of news groups) based on last column values
total_docs = train.shape[0] # Number of documents in trianing set
v_total = train.shape[1] - 2  # Number of unique words in training set, need to 
#drop index and class columns, so - 2
column_count = train.shape[1] # Number of columns prior to removing unneeded 
#index and target columns


def build_weights(): # Returns array of maximizing weights (w)

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
        print("Class:" + str(dataframes[dataframe].iloc[0]['class_id']))
        print("Size: " + str(len(dataframes[dataframe])))
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


weights = build_weights()