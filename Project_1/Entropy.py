import pandas as pd
import numpy as np


def entropy(a):
  a_total = a[0] + a[1]
  #print(a_total)
  p1 = (a[0] / a_total)
  #print(p1)
  p2 = (a[1] / a_total)
  #print(p2)
  return -(p1 * (np.log2(p1))) - (p2 * (np.log2(p2)))

a = [4, 6]

print(entropy(a))