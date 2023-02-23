import pandas as pd
import numpy as np

#returns gini impurity score (less is best).
def gini(a):
    p = 0
    a_sum = sum(a)
    for i in range(len(a)):
        p += np.square(a[i] / a_sum)
    return 1 - p

a = [4,1]
print(gini(a))