import pandas as pd
from scipy.sparse import csr_matrix
from scipy import sparse
import numpy as np
from numpy import genfromtxt
from numpy import asarray
from numpy import savetxt
from numpy import loadtxt
import math
import matplotlib.pyplot as plt
from collections import Counter

'''
Adjust these global variables for specific training set:
'''
run_loop = False # Set to false so that results and accuracy is found for the single beta value

plot_results = False # Set to true to plot results (single beta value will plot true target vs
# predicted target)

true_targets = True # Set to true if the test data set has the true target values in the last column
'''
End of global variables  
'''

'''
get_results returns a list of predicted target values as well as a list of the 
true target values if the test data provides it
'''
def get_predictions(test_np, lh_np, prior, true_targets_exist):

    # print()
    # print(lh_np[0][0:4])
    result = []
    y = None
    # print(test_np.shape)
    if true_targets_exist:
        y = test_np[:, lh_np.shape[1]] # Creating list from last column (news groups)
        # Delete last column (news group)
        test_np = np.delete(test_np, lh_np.shape[1] - 1, 1)

    test_np_len = len(test_np)

    for i in range(test_np_len):
        a = test_np[i]
        b = (a * lh_np)
        e = []
        for j in range(lh_np.shape[0]):
            x = b[j] # Setting x to list of MAP values for current news group
            x = np.log2(x, out=np.zeros_like(x), where=(x!=0)) # Taking the log_base2 of all x (MAP) values
            c = np.dot(a,x) # Taking dot product of test word set (test_np[i]) and x 
            d = (np.log2(prior[j]) + c) # Adding c to to log_base2 of current news group's prior 
            e.append(d) 
        # print(e)
        # print()
        result.append(np.argmax(e))
        del e

    result = np.array(result)
    for i in range(len(result)):
        result[i] = result[i] + 1

    # print("y values")
    # print(y.shape)
    # print(y)

    # print("results values")
    # print(result.shape)
    # print(result)
    return result, y

'''
get_accuracy returns the number of correctly predicted targets over the total number of predicted 
targets to get the precent correct
'''
def get_accuracy(test_np, result, y):
    test_np_len = len(test_np)
    accuracy_count = 0
    for i in range(test_np_len):
        if y[i] == result[i]:
            accuracy_count += 1

    print(accuracy_count / test_np_len)

    return accuracy_count / test_np_len


'''
Body of test.py
'''
# Loading MAP and MLE values from latest training run
lh_np = loadtxt('Project_2/lh_np.csv', delimiter=',')
# print(lh_np.shape)

prior = lh_np[:, lh_np.shape[1] - 1] # List of priors for each news group
lh_np = np.delete(lh_np, lh_np.shape[1] - 1, 1) # List of MAP values for each news group
# print(lh_np.shape)

# Loading the testing data set 
test_sparse = sparse.load_npz('Project_2/csr_test.csv_lg.npz')
test = sparse.csr_matrix(test_sparse)
test_np = csr_matrix.toarray(test)
test_np = np.delete(test_np, 0, 1)  # Delete 1st column (index)

if not run_loop:
    # Getting predictions for the test data 
    result,y = get_predictions(test_np, lh_np, prior, true_targets)
    savetxt('Project_2/NB_result.csv', result, delimiter=',', header='predictions', 
           fmt ='% s')
    if true_targets:
        # Getting the accuracy of the test data set's predictions
        accuracy = get_accuracy(test_np, result, y)
     
        if plot_results:

            # Creating scatter plot of results 
            c = Counter(zip(result.copy(),y.copy()))
            # Creating list to increase size of points with greater frequency 
            s = [10*c[(results,yy)] for results,yy in zip(result,y)] 
            # Creating actual plot and showing it
            plt.scatter(result, y, s=s)
            plt.locator_params(axis="both", integer=True, tight=True)
            plt.ylim(0,lh_np.shape[0] + 1)
            plt.xlim(0,result.shape[0] + 1)
            plt.show()

else: 
    beta_test_results = []

    for i in range(0,50):
        beta_MAP_MLE = loadtxt('Project_2/Diff_Beta_Values/beta_'+str(i)+'.csv', delimiter=',')
        beta_prior = beta_MAP_MLE[:, beta_MAP_MLE.shape[1] - 1] # List of priors for each news group
        # List of MAP values for each news group
        beta_MAP_MLE = np.delete(beta_MAP_MLE, beta_MAP_MLE.shape[1] - 1, 1)

        # print()
        # print(beta_MAP_MLE[0][0:4])

        beta_result,beta_y = get_predictions(test_np, beta_MAP_MLE, beta_prior, true_targets)

        if true_targets:
            beta_accuracy = get_accuracy(test_np, beta_result, beta_y)

            beta_test_results.append(beta_accuracy)

    if plot_results and true_targets:
        # print(beta_test_results)
        x = np.linspace(0.00001,1,50)
        # print(x)
        plt.plot(x,beta_test_results)
        plt.xscale("log")
        plt.show()



