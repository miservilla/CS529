import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import scipy

def chi(p, a, CI):
    cv = scipy.stats.chi2.ppf(CI, len(a) - 1)
    dfree = len(a) - 1
    c = 0
    p_tot = p[0] + p[1]
    for i in a:
        a_tot = i[0] + i[1]
        a_exp0 = a_tot * (p[0] / p_tot)
        a_exp1 = a_tot * (p[1] / p_tot)
        c += np.square(i[0] - a_exp0) / a_exp0 + \
        np.square(i[1] - a_exp1) / a_exp1
    print(c)
    print(cv)
    print(c > cv)


p = [9, 5]
a = [[6, 2], [3, 3]]
CI =  0.95
chi(p, a, CI)
    