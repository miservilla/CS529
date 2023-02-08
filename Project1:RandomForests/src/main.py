import pandas as pd
import numpy as np
import Tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import scipy


# Training data set
df_train = pd.read_csv('Project1:RandomForests/src/bill_authentication.csv')
# print(df_train.shape)
df_train, df_testing = train_test_split(df_train, test_size=0.2)
# df_testing = pd.read_csv('DecisionTree/Test.csv')  # Testing data set
df_testing = df_testing.sample(frac=1).reset_index(drop=True)
# print(len(df_train), len(df_testing))

# Removing column names that aren't attributes from training set
attributes = list(df_train.columns.values)
del attributes[:1]
# del attributes[4:]  
# print(attributes)

# Setting binary target column from provided target column 
target = LabelEncoder()
df_train['target'] = target.fit_transform(df_train['Class'])
print(df_testing)
df_testing['test_target'] = target.fit_transform(df_testing['Class'])
# print(df_testing)
testing = df_testing[['test_target']]
test = []
for i in range(len(testing)):
    test.append(testing.iloc[i, 0])
print(test)
# print(df['target'])

def build_binary_value_list(value_count) -> list:
    count = []
    try:
        count.append(value_count[1])
    except:
        count.append(0)
    try:
        count.append(value_count[0])
    except:
        count.append(0)
    return count

def entropy(a): 
  a_total = a[0] + a[1]
  #print(a_total)
  p1 = (a[0] / a_total)
  #print(p1)
  p2 = (a[1] / a_total)
  #print(p2)
  if p1 == 0 or p2 == 0:
    return 0
  else:
    return -(p1 * (np.log2(p1))) - (p2 * (np.log2(p2)))

def IG_entropy(attribute, labels: list, df, tc):
    IG = entropy(tc)
    total = df.shape[0]
    for label in labels:
        babyFrame = df.loc[df[attribute] == label]
        babytotal = babyFrame.shape[0]
        ltc = babyFrame['target'].value_counts()
        count = build_binary_value_list(ltc)
        IG = IG - ((babytotal/total)*entropy(count))
    return IG

def gini(a):
    p = 0
    a_sum = sum(a)
    for i in range(len(a)):
        p += np.square(a[i] / a_sum)
    return 1 - p

def IG_gini(attribute, labels: list, df, tc):
    IG = gini(tc)
    total = df.shape[0]
    for label in labels:
        babyFrame = df.loc[df[attribute] == label]
        babytotal = babyFrame.shape[0]
        ltc = babyFrame['target'].value_counts()
        count = build_binary_value_list(ltc)
        IG = IG - ((babytotal/total)*gini(count))
    return IG

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
    return(c > cv)

def chi_info_finder(attribute, df, labels):
    parent_vc = df['target'].value_counts()
    parent_count = build_binary_value_list(parent_vc)
    label_counts = []
    for label in labels:
        babyFrame = df.loc[df[attribute] == label]
        ltc = babyFrame['target'].value_counts()
        count = build_binary_value_list(ltc)
        label_counts.append(count)
    # print(parent_count, label_counts)
    return chi(parent_count, label_counts, 0.95)


def build_binary_DT(attributes, df, DT_type, parent) -> Tree.DTree:

    if len(df['target'].unique()) == 1: 
        # print("Single Value: returning LEAF of " + str(df['target'].unique()[0]))
        return Tree.DTree({'Parent_Branch':parent,'Leaf':df['target'].unique()[0]},None, True)

    if len(attributes) == 0:
        values = df['target'].value_counts()
        if values[0]>values[1]:
            # print("No more attributes: returning LEAF of " + str(0))
            return Tree.DTree({'Parent_Branch':parent,'Leaf': 0},None, True)
        else:  
            # print("No more attributes: returning LEAF of " + str(1))
            return Tree.DTree({'Parent_Branch':parent,'Leaf': 1},None, True)
        
    highest = 0
    highest_att = ""

    tc = df['target'].value_counts()
    target_count = []
    try:
        target_count.append(tc[1])
    except:
        target_count.append(0)
    try:
        target_count.append(tc[0])
    except:
        target_count.append(0)
    
    #Finding I.G. for each attribute according to selected method
    for attribute in attributes: 
        labels = df[attribute].unique()

        if DT_type == "entropy":
           IG = IG_entropy(attribute, labels, df, target_count)
           chi2 = chi_info_finder(attribute, df, labels)
           if IG > highest:
            highest = IG
            highest_att = attribute

        elif DT_type == "gini":
            IG = IG_gini(attribute, labels, df, target_count)
            chi2 = chi_info_finder(attribute, df, labels)
            if IG > highest:
                highest = IG
                highest_att = attribute

        # elif DT_type == "ME": 
            
    # print("Highest I.G. attribute is " + highest_att)

    highest_labels = df[highest_att].unique() 
    attributes.remove(highest_att)

    lower_branch = []

    for label in highest_labels:
        sub_df = df.loc[df[highest_att] == label]
        # print("Going down branch " + str(label))
        # print(sub_df)
        lower_branch.append(build_binary_DT(attributes, sub_df, DT_type, label))

    dic = {}
    dic['Parent_Branch'] = parent
    dic['Attribute'] = highest_att
    dic['Labels'] = highest_labels

    return Tree.DTree(dic,lower_branch)


def print_tree(tree: Tree.DTree, level:int):

    print("Tree Level: " + str(level))
    print(tree.node) 

    for branch in tree.attributes:
        print_tree(branch, level+1)

def traverse_tree(tree: Tree.DTree, iter):
    if not tree.isLeaf:
        current_att = tree.node['Attribute']
        iter_att_label = iter[current_att]
        # print(iter_att_label)
        for branch in tree.attributes:
            if branch.node['Parent_Branch'] == iter_att_label:
                return traverse_tree(branch, iter)
    else:
        return tree.node['Leaf']

        
def predict(tree: Tree.DTree, testing):
    predictions = []
    for i in range(len(testing)):
        # print("Showing attributes for row " + str(i) + "**********************")
        # print(testing.iloc[i])
        predictions.append(traverse_tree(tree, testing.iloc[i]))
    score = accuracy_score(test, predictions)
    print(score)
    print(predictions)
    return predictions

def plant_forest(attributes: list, df, DT_type: str, forest_size: int, sample_size: int):
    forest = []
    for i in range(forest_size):
        sub_df = df.sample(n=sample_size)
        forest.append(build_binary_DT(attributes, sub_df, DT_type, "root"))
    return forest


# entropy_Tree = build_binary_DT(attributes, df, "entropy", "root")
# print_tree(entropy_Tree, 1)

gini_Tree = build_binary_DT(attributes, df_train, "gini", "root")
# print_tree(gini_Tree, 1)
predict(gini_Tree, df_testing)