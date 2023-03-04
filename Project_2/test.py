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

lh_np = loadtxt('Project_2/lh_np.csv', delimiter=',')
print(lh_np.shape)

prior = lh_np[:, lh_np.shape[1] - 1] # List of priors for each news group
# List of MAP values for each news group
lh_np = np.delete(lh_np, lh_np.shape[1] - 1, 1)
# print(lh_np.shape)

test_sparse = sparse.load_npz('Project_2/csr_test.csv.npz')
test = sparse.csr_matrix(test_sparse)
test_np = csr_matrix.toarray(test)
test_np = np.delete(test_np, 0, 1)  # Delete 1st column (index)

# product = [[0 for x in range(20)] for y in range(2400)]
# product = np.array(product)

result = []
print(test_np.shape)
y = test_np[:, lh_np.shape[1]] # Creating list from last column (news groups)
# Delete last column (news group)
test_np = np.delete(test_np, lh_np.shape[1] - 1, 1)

test_np_len = len(test_np)


# ************************* I changed this part to use the 'classify' agorithm 
#                           found under the Naive Bayes implementation in the pdf
#                           for project 2, it now gets 0.855                      ********************************************************
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
    result.append(np.argmax(e))
    del e

result = np.array(result)
for i in range(len(result)):
    result[i] = result[i] + 1

print("y values")
print(y.shape)
print(y)

print("results values")
print(result.shape)
print(result)

accuracy_count = 0
for i in range(test_np_len):
    if y[i] == result[i]:
        accuracy_count += 1

print(accuracy_count / test_np_len)


# Creating scatter plot of results 
# Y-axis: True values representing news groups
# X-axis: Predicted values representing news groups
c = Counter(zip(result.copy(),y.copy()))
# Creating list to increase size of points with greater frequency 
s = [10*c[(results,yy)] for results,yy in zip(result,y)] 
# Creating actual plot and showing it
plt.scatter(result, y, s=s)
plt.locator_params(axis="both", integer=True, tight=True)
plt.ylim(0,lh_np.shape[0] + 1)
plt.xlim(0,result.shape[0] + 1)
plt.show()