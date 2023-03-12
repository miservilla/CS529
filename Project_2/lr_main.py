import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix
from scipy import sparse
import numpy as np
from math import log

'''
total_docs = number of examples (m in PDF)
unique_targets = number of classes (in this context number of news groups)
v_total = number of attributes -> unique words in training set (n in PDF)
aprox_prediction = exp(WX^T) used in place of P(Y|W,X)
delta = 1 or 0 depending on if Y^l = yj or Y^l !+ yj

'''


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
Ys = train[:,column_count-1] # A m by 1 vector of true classifications
# print(Ys)

def multiply_weights(weights, Xs): 
    return weights*Xs

def set_weights(Xs_length):
    return np.random.rand(1, Xs_length)

def update_weights(current_weights, learning, penalty, aprox_prediction, delta, Xs):
    
    length = current_weights.shape[1]
    # new_weights = np.zeros(shape = (1,length))

    d_ap = delta-aprox_prediction
    mult_Xs = d_ap*Xs
    mult_Xs_sum = mult_Xs.sum(axis=0)
    pen_Ws = penalty * current_weights
    sum_pen = mult_Xs_sum-pen_Ws
    mult_learn = learning * sum_pen
    new_weights = current_weights + mult_learn
    # print(new_weights)
    
    return new_weights

def get_aprox_prediction(weights, Xs): # Returns the P(Y|W,X) from in the PDF
    predict = np.exp(np.dot(weights,np.transpose(Xs).toarray()))
    predict[unique_targets-1,:] = 1
    pt = np.transpose(predict)
    predict = np.transpose(pt/pt.sum(axis=1)[:,None])
    # print(predict)
    return predict

def get_likelihood(Yj, weights, Xs):
    return int(Yj)*sum(weights*Xs) - log(1 + np.exp(sum(weights*Xs)))

def get_delta():
    delta_matrix = np.zeros(shape=(unique_targets, total_docs))
    for i in range(delta_matrix.shape[0]):
        for j in range(delta_matrix.shape[1]):
            if i+1 == Ys[j,0]:
                delta_matrix[i,j] = 1

    # print(delta_matrix)
    return delta_matrix


def minimize_likelihood(last_likelihood, current_likelihood, last_weights, current_weights : dict, Xs, Xjs, delta):
    print()
    print("****************************************************************")

    likelihood = 0

    length = len(current_weights)
    last_weights = current_weights.copy()

    weights_matrix = np.zeros(shape=(length,current_weights["1"].shape[1]))

    for i in range(length): # Creating matrix of current weights for P(Y|W,X) aproximation
        cur_class = str(i + 1)
        weights_matrix[i] = np.asarray(current_weights[cur_class])

    # print(weights_matrix)

    predict = get_aprox_prediction(weights_matrix, Xs)
    # print(predict)

    for i in range(length): # Updating weights for each class
        cur_class = str(i + 1)
        # print()
        # print("Class: " + str(cur_class))
        # print(last_weights[cur_class])
        current_weights[cur_class] = update_weights(np.asarray(current_weights[cur_class]), 0.01, 0.1, predict, delta, Xs)

        likelihood = likelihood + get_likelihood(cur_class, np.squeeze(np.asarray(current_weights[cur_class])), np.squeeze(np.asarray(Xjs[cur_class])))

    print()
    print("Likelihood: " + str(likelihood))

    return current_likelihood, likelihood, last_weights, current_weights

    

def build_weights(): # Returns array of maximizing weights (w)

    # Getting matrix of Xs 
    Xs = train.copy()
    Xs[:,0] = np.ones(total_docs)
    Xs = Xs[:, :-1]

    # Getting matrix of delta values
    delta = get_delta()

    df = pd.DataFrame(columns=['index', 'class_id'])

    for i in range(total_docs):
        df.loc[len(df.index)] = [i, train[i, column_count - 1]] # df starts at index 0

    dataframes = {} # Dictionary of dataframes with newsgroups indexes 
    for i in range(1, unique_targets+1):
        dataframes["df_{0}".format(i)] = df[df['class_id'] == i]

    wrd_per_NG = csr_matrix((unique_targets,v_total+1), dtype=int)

    sparse_classes = {} # Dictionary of sparse matrix for each newsgroup
    weights = {} # Dictionary of wieghts with index associated with news group
    count = 1

    likelihood = 0

    for dataframe in dataframes:
        print()
        print("Class:" + str(dataframes[dataframe].iloc[0]['class_id']))
        print("Size: " + str(len(dataframes[dataframe])))
        print(dataframes[dataframe])
        # yk_docs_cnt = len(dataframes[dataframe])
        index = str(dataframes[dataframe].iloc[0]['class_id'])

        sparse_classes["{0}".format(index)] = csr_matrix((1,column_count), dtype=int)

        for row in dataframes[dataframe]['index']:
            sparse_classes[index] = train[row].toarray() + sparse_classes[index]

        # Deleting last column as they are news groups (not words)
        # and setting all of first column to 1 to use with weight W0 
        sparse_classes[index][0,0] = 1
        sparse_classes[index] = np.delete(sparse_classes[index], v_total+1)
        # print(sparse_classes[index][0,:])
        # print(sparse_classes[index].shape)

        
        weights["{0}".format(index)] = set_weights(sparse_classes[index].shape[1])
        # print(weights[index].shape)

        likelihood = likelihood + get_likelihood(index, np.squeeze(np.asarray(weights[index])), np.squeeze(np.asarray(sparse_classes[index])))

    print()
    print("Likelihood: " + str(likelihood))

    last_likelihood = likelihood+1
    current_likelihood = likelihood
    last_weights = None
    current_weights = weights

    while last_likelihood > current_likelihood:
    
        last_likelihood, current_likelihood, last_weights, current_weights = minimize_likelihood(last_likelihood, current_likelihood, last_weights, current_weights, Xs, sparse_classes, delta)

    return last_weights


        

weights = build_weights()

print(weights)