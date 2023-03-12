import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix
from scipy import sparse
import numpy as np

'''
total_docs = number of examples (m in PDF)
unique_targets = number of classes (in this context number of news groups)
v_total = number of attributes -> unique words in training set (n in PDF)
aprox_prediction = exp(WX^T) used in place of P(Y|W,X)
delta = 1 or 0 depending on if Y^l = yj or Y^l !+ yj

'''


# Loading training data 
train_sparse = sparse.load_npz(
    'Project_2/csr_train.csv_sm.npz')


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
learning = 0.5
penalty = 0.01

# Getting matrix of Xs 
Xs = train.copy()
Xs[:,0] = np.ones(total_docs)
Xs = Xs[:, :-1]

# Getting matrix of delta values
def get_delta():
    delta_matrix = np.zeros(shape=(unique_targets, total_docs))
    for i in range(delta_matrix.shape[0]):
        for j in range(delta_matrix.shape[1]):
            if i+1 == Ys[j,0]:
                delta_matrix[i,j] = 1

    # print(delta_matrix)
    return delta_matrix
delta = get_delta()

def set_weights(Xs_length): # Initializing weights with random floats
    return np.random.rand(1, Xs_length)

# def update_weights(current_weights, learning, penalty, aprox_prediction):    
#     return current_weights + (  learning * (((delta - aprox_prediction) * Xs).sum(axis=0) - (penalty * current_weights)))

def get_aprox_prediction(weights): # Returns the P(Y|W,X) from in the PDF
    predict = np.exp(np.dot(weights,np.transpose(Xs).toarray()))
    predict[unique_targets-1,:] = 1
    # print(predict)
    pt = np.transpose(predict)
    predict = np.transpose(pt/pt.sum(axis=1)[:,None])
    # print(predict)
    return predict

def get_likelihood(Yl, weights, Xljs):
    return int(Yl)*np.sum(weights*Xljs) - np.log(1 + np.exp(np.float128( np.sum(weights*Xljs)))) # Conditional likelihood equation from PDF


def minimize_likelihood(last_likelihood, weights_matrix):
    print()
    print("****************************************************************")

    weights_matrix2 = np.zeros(shape=(unique_targets,v_total+1))

    predict = get_aprox_prediction(weights_matrix)
    # print(predict)

    for i in range(unique_targets): # Updating weights for each class
        # print()
        # print("Class: " + str(i+1))
        # print(last_weights[cur_class])
        weights_matrix2[i] = weights_matrix[i] + (  learning * (((delta - predict) * Xs).sum(axis=0) - (penalty * weights_matrix[i])))

        # print(weights_matrix)
        # print(weights_matrix2)

    likelihood = 0

    for j in range(total_docs):
        likelihood = likelihood + get_likelihood(Ys[j,0],weights_matrix2[Ys[j,0]-1,:], Xs[j,:].toarray())

    print()
    print("Likelihood: " + str(likelihood))

    if last_likelihood > likelihood:
        return likelihood, weights_matrix2
    else:
        return last_likelihood, weights_matrix

    

def build_weights(): # Returns array of maximizing weights (w)

    weights_matrix = np.zeros(shape=(unique_targets,v_total+1))

    for i in range(unique_targets): # Creating matrix of weights
        weights_matrix[i] = set_weights(v_total+1)
    
    likelihood = 0

    for j in range(total_docs):
        likelihood = likelihood + get_likelihood(Ys[j,0],weights_matrix[Ys[j,0]-1,:], Xs[j,:].toarray())

    print()
    print("Likelihood: " + str(likelihood))

    last_likelihood = likelihood+1
    current_likelihood = likelihood

    while last_likelihood > current_likelihood:
        last_likelihood = current_likelihood.copy()
        current_likelihood, weights_matrix = minimize_likelihood(last_likelihood, weights_matrix)

    return weights_matrix


      

weights = build_weights()

print(weights)
print(get_aprox_prediction(weights))