
import pandas as pd
from scipy.sparse import csr_matrix
from scipy import sparse
import numpy as np
from numpy import genfromtxt
from numpy import asarray
from numpy import savetxt
from numpy import loadtxt
import math




train_sparse = sparse.load_npz(
    '/home/michaelservilla/CS529/Project_2/csr_train.csv_lg.npz')

target_column = 61189
not_an_attribute = [0, 61189]

#training set without class columns
X = sparse.csr_matrix(train_sparse)
df_train = pd.DataFrame.sparse.from_spmatrix(X)
# print(df_train)

attributes = list(df_train.columns.values)
for item in not_an_attribute:
    attributes.remove(item)

def build_binary_value_list(value_count) -> list:
    count = []
    try:
        count.append(value_count[1])
    except:
        count.append(0)
    try:
        count.append(value_count[2])
    except:
        count.append(0)
    try:
        count.append(value_count[3])
    except:
        count.append(0)
    try:
        count.append(value_count[4])
    except:
        count.append(0)
    try:
        count.append(value_count[5])
    except:
        count.append(0)
    try:
        count.append(value_count[6])
    except:
        count.append(0)
    try:
        count.append(value_count[7])
    except:
        count.append(0)
    try:
        count.append(value_count[8])
    except:
        count.append(0)
    try:
        count.append(value_count[9])
    except:
        count.append(0)
    try:
        count.append(value_count[10])
    except:
        count.append(0)
    try:
        count.append(value_count[11])
    except:
        count.append(0)
    try:
        count.append(value_count[12])
    except:
        count.append(0)
    try:
        count.append(value_count[13])
    except:
        count.append(0)
    try:
        count.append(value_count[14])
    except:
        count.append(0)
    try:
        count.append(value_count[15])
    except:
        count.append(0)
    try:
        count.append(value_count[16])
    except:
        count.append(0)
    try:
        count.append(value_count[17])
    except:
        count.append(0)
    try:
        count.append(value_count[18])
    except:
        count.append(0)
    try:
        count.append(value_count[19])
    except:
        count.append(0)
    try:
        count.append(value_count[20])
    except:
        count.append(0)
    return count


# 
# Returns entropy value
# 
def entropy(a: list): 
  a_total = a[0] + a[1] + a[2] + a[3] + a[4] + a[5] + a[6] + a[7] + a[8] + a[9] + a[10] + a[11] +\
            a[12] + a[13] + a[14] + a[15] + a[16] + a[17] + a[18] + a[19]
  if a_total == 0:
    a_total = 1
  #print(a_total)
  p1 = (a[0] / a_total)
  p2 = (a[1] / a_total)
  p3 = (a[2] / a_total)
  p4 = (a[3] / a_total)
  p5 = (a[4] / a_total)
  p6 = (a[5] / a_total)
  p7 = (a[6] / a_total)
  p8 = (a[7] / a_total)
  p9 = (a[8] / a_total)
  p10 = (a[9] / a_total)
  p11 = (a[10] / a_total)
  p12 = (a[11] / a_total)
  p13 = (a[12] / a_total)
  p14 = (a[13] / a_total)
  p15 = (a[14] / a_total)
  p16 = (a[15] / a_total)
  p17 = (a[16] / a_total)
  p18 = (a[17] / a_total)
  p19 = (a[18] / a_total)
  p20 = (a[19] / a_total)

  if p1 == 0 or p2 == 0 or p3 == 0 or p4 == 0 or p5 == 0 or p6 == 0 or p7 == 0 or p8 == 0 \
    or p9 == 0 or p10 == 0 or p11 == 0 or p12 == 0 or p13 == 0 or p14 == 0 or p15 == 0 or \
    p16 == 0 or p17 == 0 or p18 == 0 or p19 == 0 or p20 == 0:
    return 0
  else:
    return -(p1 * (np.log2(p1))) - (p2 * (np.log2(p2))) - (p3 * (np.log2(p3))) -\
            (p4 * (np.log2(p4))) - (p5 * (np.log2(p5))) - (p6 * (np.log2(p6))) -\
            (p7 * (np.log2(p7))) - (p8 * (np.log2(p8))) - (p9 * (np.log2(p9))) -\
            (p10 * (np.log2(p10))) - (p11 * (np.log2(p11))) - (p12 * (np.log2(p12))) -\
            (p13 * (np.log2(p13))) - (p14 * (np.log2(p14))) - (p15 * (np.log2(p15))) -\
            (p16 * (np.log2(p16))) - (p17 * (np.log2(p17))) - (p18 * (np.log2(p18))) -\
            (p19 * (np.log2(p19))) - (p20 * (np.log2(p20)))


# 
# Caculates IG using entropy
# 
def IG_entropy(attribute, labels: list, df: pd.DataFrame, tc: list):
    IG = entropy(tc)
    print("orig tc : " + str(tc))
    print("orig IG: " + str(IG))
    total = df.shape[0]
    for label in labels:
        babyFrame = df.loc[df[attribute] == label]
        babytotal = babyFrame.shape[0]
        ltc = babyFrame[target_column].value_counts()
        count = build_binary_value_list(ltc)
        # print(count)
        IG = IG - ((babytotal/total)*entropy(count))
    return IG


def get_IG(attributes: list, df):
    tc = df[target_column].value_counts()
    target_count = []
    try:
        target_count.append(tc[1])
    except:
        target_count.append(0)
    try:
        target_count.append(tc[2])
    except:
        target_count.append(0)
    try:
        target_count.append(tc[3])
    except:
        target_count.append(0)
    try:
        target_count.append(tc[4])
    except:
        target_count.append(0)
    try:
        target_count.append(tc[5])
    except:
        target_count.append(0)
    try:
        target_count.append(tc[6])
    except:
        target_count.append(0)
    try:
        target_count.append(tc[7])
    except:
        target_count.append(0)
    try:
        target_count.append(tc[8])
    except:
        target_count.append(0)
    try:
        target_count.append(tc[9])
    except:
        target_count.append(0)
    try:
        target_count.append(tc[10])
    except:
        target_count.append(0)
    try:
        target_count.append(tc[11])
    except:
        target_count.append(0)
    try:
        target_count.append(tc[12])
    except:
        target_count.append(0)
    try:
        target_count.append(tc[13])
    except:
        target_count.append(0)
    try:
        target_count.append(tc[14])
    except:
        target_count.append(0)
    try:
        target_count.append(tc[15])
    except:
        target_count.append(0)
    try:
        target_count.append(tc[16])
    except:
        target_count.append(0)
    try:
        target_count.append(tc[17])
    except:
        target_count.append(0)
    try:
        target_count.append(tc[18])
    except:
        target_count.append(0)
    try:
        target_count.append(tc[19])
    except:
        target_count.append(0)
    try:
        target_count.append(tc[20])
    except:
        target_count.append(0)

    all_IG = []

    for attribute in attributes: 
        print("Working on:" + str(attribute))
        labels = df[attribute].unique()
        IG = IG_entropy(attribute, labels, df, target_count)
        print("I.G. is: " + str(IG))
        all_IG.append(IG)

    print(all_IG)

    return all_IG

IG_all = np.array(get_IG(attributes, df_train))

max_100 =  np.argpartition(IG_all,-100)[-100:]
max_100 = max_100[np.argsort(IG_all[max_100])]
max_100 = max_100[::-1]

savetxt('Project_2/max_100_words.csv', max_100, delimiter=',', 
        fmt ='% s')