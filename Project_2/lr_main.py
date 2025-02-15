import pandas as pd
from scipy.sparse import csr_matrix, lil_matrix
from scipy import sparse
import numpy as np


''' Global Variables Start: Adjust these global values for specific training set '''

# Loading training data 
train_sparse = sparse.load_npz(
    'Project_2/csr_train.csv_lg.npz')

# Traing data as sparse matrix
train = sparse.csr_matrix(train_sparse) 
print(train.shape)

# Number of possible targets (number of news groups) based on last column values
unique_targets = train[np.argmax(train[:, -1]), -1] 

# Number of documents in trianing set
total_docs = train.shape[0] 

# Number of unique words in training set, need to drop index and class columns, so - 2
v_total = train.shape[1] - 2  

# Number of columns prior to removing unneeded index and target columns
column_count = train.shape[1]

# A v_total by 1 vector of true classifications
y = train[:,column_count-1].toarray() 
# print(y)

# Learning rate used to update the weights
learning = 0.01

# Penalty applied to the weights while updating them
penalty = 0.01

# Number of times the weights get updated
max_iterations = 1

# Setting matrix for X where the first column is all 1's to deal with W0
# x = sparse.csr_matrix(sparse.load_npz(
#     'Project_2/csr_train.csv_lg_reduced.npz'))
x = train.copy()
x = x[:, :-1]
x = x/1000
x[:,0] = np.ones(total_docs)

# Setting matrix for delta where if Yj = Yi the value is 1, otherwie 0
delta = np.zeros(shape=(unique_targets, total_docs))
for i in range(delta.shape[0]):
    for j in range(delta.shape[1]):
        if i == y[j,0]-1:
            delta[i,j] = 1
print(delta)

''' Global Variables End '''

# Initializing weights with 0's
def set_weights(x_length) -> np.ndarray: 
    weights = np.zeros(shape=(unique_targets,v_total+1))
    return weights

def print_prediction(prediction):
    print("Prediction: ------------------------------------------------")
    rows = prediction.shape[0]
    cols = prediction.shape[1]
    for i in range(rows):
        for j in range(cols):
            print(str(prediction[i][j])[:7], end=" ")
        print()


# Returns the predicted likelihood for each example that it is in a particular class -> P(Y|W,X) 
def calculate_prediction(weights, xes) -> np.ndarray: 
    prediction = np.exp(weights*np.transpose(xes))
    prediction[unique_targets-1,:] = 1
    pt = np.transpose(prediction)
    prediction = np.transpose(pt/pt.sum(axis=1)[:,None])

    # print_prediction(prediction)
    print(prediction)

    return prediction

# Returns the conditional likelihood value for the input weights 
def calculate_likelihood(weights):
    likelihood = 0

    for j in range(total_docs):
        # Target value for current example
        target = y[j]-1
        # print(target)
        # Vector of X values for current example
        xj = x[j,:].toarray()
        # print(xj[0])
        # vector of W (weight) values for current example's target
        w = weights[target,:]
        # print(w[0])
        # Dot product of w and x
        dot = np.dot(w[0],xj[0])
        # Updating likelihood
        likelihood = likelihood + ((y[j]* dot)- np.log(1 + np.exp(dot)))

    return likelihood


def maximize_likelihood(last_likelihood, last_weights):

    new_weights = np.zeros(shape=(unique_targets,v_total+1))

    prediction = calculate_prediction(last_weights,x)

    loss = delta - prediction

    # print()
    # print("Loss: ")
    # print(loss)

    loss_x = loss*x

    # print()
    # print("x: ")
    # print(x)

    # print()
    # print("Loss x: ")
    # print(loss_x)

    pen_weights = penalty * last_weights

    # print()
    # print("penalized weights: ")
    # print(pen_weights)


    new_weights = last_weights + (  learning * (loss_x - pen_weights))

    # print()
    # print("new weights: ")
    # print(new_weights)

    new_likelihood = calculate_likelihood(new_weights)

    if last_likelihood < new_likelihood:
        print()
        print("Likelihood: " + str(new_likelihood))
        return new_likelihood, new_weights
    else:
        print()
        print("Likelihood: " + str(last_likelihood))
        return last_likelihood, last_weights

    
# Returns matrix of weights that maximize the conditional likelihood function
def find_weights(): 

    ''' Dataframe dictionary not needed in final code, only used to see what class each example is in '''

    # df = pd.DataFrame(columns=['index', 'class_id'])

    # for i in range(total_docs):
    #     df.loc[len(df.index)] = [i, train[i, column_count - 1]] 
    
    # # Dictionary of dataframes with newsgroups indexes
    # dataframes = {}  
    # for i in range(1, unique_targets+1):
    #     dataframes["df_{0}".format(i)] = df[df['class_id'] == i]

    # for dataframe in dataframes:
    #     print()
    #     print("Class:" + str(dataframes[dataframe].iloc[0]['class_id']))
    #     print("Size: " + str(len(dataframes[dataframe])))
    #     print(dataframes[dataframe])

    ''' End of code used for dataframe dictionary '''


    weights = set_weights(v_total+1)
    
    likelihood = calculate_likelihood(weights)
    
    print()
    print("Likelihood: " + str(likelihood))

    for iter in range(max_iterations):
        likelihood, weights = maximize_likelihood(likelihood, weights)
        iter += 1


    return weights


def class_prediction(prediction, target):
    predicted_class = np.zeros(target.shape[0], dtype=int)
    for i in range(prediction.shape[1]):
        tmp = prediction[:,i]
        location = np.unravel_index(np.argmax(tmp), tmp.shape)
        predicted_class[i] = location[0] + 1

    return predicted_class

def print_class_prediction(predicted_class, target):
    print()
    print("True target vs Predicted target: ----------------------------")
    print(target)
    print(predicted_class)

def get_accuracy(predicted_class, target):
    length = len(predicted_class)
    correct = 0
    for i in range(length):
        if predicted_class[i] == target[i]:
            correct += 1
    return correct/length


weights = find_weights()
final_predictions = calculate_prediction(weights,x)
predicted_class = class_prediction(final_predictions,np.transpose(y)[0])

print("****************************************************************")
# print(weights)
# print()
# print_prediction(final_predictions)
print_class_prediction(predicted_class, np.transpose(y)[0])
print(get_accuracy(predicted_class, np.transpose(y)[0]))
print("****************************************************************")
