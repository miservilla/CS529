import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder

s_target = LabelEncoder()
S = pd.read_csv('/home/michaelservilla/CS529/DecisionTree/Test.csv')

def entropy(a):
  a_total = a[0] + a[1]
  p1 = (a[0] / a_total)
  p2 = (a[1] / a_total)
  return -(p1 * (np.log2(p1))) - (p2 * (np.log2(p2)))

#returns gini impurity score (less is best).
def gini(a):
    p = 0
    a_sum = sum(a)
    for i in range(len(a)):
        p += np.square(a[i] / a_sum)
    return 1 - p


def gain(attribute, S):
  SGB1 = S.groupby(S.columns[attribute]).size(
  ).reset_index(name="Count")  # total count
  SGB2 = S.groupby(S.columns[attribute])[
      S.columns[-1]].sum().reset_index(name="Count")  # positive count
  for i in range(len(SGB1)):
    x = SGB1.iloc[i, 1]
    y = SGB1.iloc[i, 1] - SGB2.iloc[i, 1]
    if x == y or y == 0:  # To determine if leaf node
      if x == y:
        print("%s is a NEGATIVE leaf node!" % SGB1.iloc[i, 0])
      else:
        print("%s is a POSITIVE leaf node!" % SGB1.iloc[i, 0])
  a = [target, SGB1, SGB2]
  gain_array = []
  target_total = target[0] + target[1]
  for i in range(len(SGB1)):
    a_total = a[1].iloc[i, 1]  # a[1]=SGB1, SGB1.iloc[0, 1]
    a_pos = a[2].iloc[i, 1]
    a_neg = a_total - a_pos
    gain_values = [a_pos, a_neg]
    if gain_values[0] != 0 and gain_values[1] != 0:
      tmp_entropy = entropy(gain_values)
    else:
      tmp_entropy = 0
    gain_array.append(-(a_total / target_total) * tmp_entropy)
  E_target = entropy(target)
  info_gain = E_target
  for i in range(len(gain_array)):
    info_gain += gain_array[i]
  print("Gain(S,%s) =" % (col_names[attribute]), info_gain)

#returns gini information gain score (higher is best).
def gini_impurity(attribute, S):
  SGB1 = S.groupby(S.columns[attribute]).size(
  ).reset_index(name="Count")  # total count
  SGB2 = S.groupby(S.columns[attribute])[
      S.columns[-1]].sum().reset_index(name="Count")  # positive count
  for i in range(len(SGB1)):
    x = SGB1.iloc[i, 1]
    y = SGB1.iloc[i, 1] - SGB2.iloc[i, 1]
    if x == y or y == 0:  # To determine if leaf node
      if x == y:
        print("%s is a NEGATIVE leaf node!" % SGB1.iloc[i, 0])
      else:
        print("%s is a POSITIVE leaf node!" % SGB1.iloc[i, 0])
  a = [target, SGB1, SGB2]
  gini_array = []
  target_total = target[0] + target[1]
  print(len(SGB1))
  for i in range(len(SGB1)):
    a_total = a[1].iloc[i, 1]  # a[1]=SGB1, SGB1.iloc[0, 1]
    a_pos = a[2].iloc[i, 1]
    a_neg = a_total - a_pos
    gini_values = [a_pos, a_neg]
    if gini_values[0] != 0 and gini_values[1] != 0:
      tmp_gini = gini(gini_values)
    else:
      tmp_gini = 0
    gini_array.append((a_total / target_total) * tmp_gini)
  S_impurity = gini(target)
  for i in range(len(gini_array)):
    S_impurity -= gini_array[i]
  print("Gini Impurity(S,%s) =" % (col_names[attribute]), S_impurity)

#Define target
S['S_target'] = s_target.fit_transform(S['PlayTennis'])

#Drop any columns not needed.
S.drop(['Day', 'PlayTennis'],axis=1, inplace=True)
# S.drop(['class'], axis=1, inplace=True)

#Set target attribute and value.
S_target = S.iloc[:, len(S.columns)-1]
target = S_target.value_counts()
target = [target.loc[0], target.loc[1]]

col_names = list(S.columns.values)
print(col_names)

for i in range(len(S.columns)-1):
  gain(i, S)

for i in range(len(S.columns)-1):
  gini_impurity(i, S)

S1 = S.loc[S['Outlook'] == ('Sunny')]
print(S1)

S_target = S1.iloc[:, len(S.columns)-1]
target = S_target.value_counts()
target = [target.loc[0], target.loc[1]]
print(target)

for i in range(len(S1.columns)-1):
  gain(i, S1)

S2 = S.loc[S['Outlook'] == ('Rain')]
print(S2)

S_target = S2.iloc[:, len(S.columns)-1]
target = S_target.value_counts()
target = [target.loc[0], target.loc[1]]
print(target)

for i in range(len(S2.columns)-1):
  gain(i, S2)
