import pandas as pd
import numpy as np
import Tree
from sklearn.preprocessing import LabelEncoder

# df = pd.read_csv('Project1:RandomForests/data/Training.csv')
df = pd.read_csv('DecisionTree/Test.csv')

attributes = list(df.columns.values)
del attributes[:1]
del attributes[4:] # removing columns that aren't attributes 
# print(attributes)

target = LabelEncoder()
df['target'] = target.fit_transform(df['PlayTennis']) #Use whatever lable target is under
# print(df['target'])

def entropy(a): # This entropy assumes a binary target
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
        count = []
        try:
            count.append(ltc[1])
        except:
            count.append(0)
        try:
            count.append(ltc[0])
        except:
            count.append(0)

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
        count = []
        try:
            count.append(ltc[1])
        except:
            count.append(0)
        try:
            count.append(ltc[0])
        except:
            count.append(0)

        IG = IG - ((babytotal/total)*gini(count))

    return IG




def build_DT(attributes, df, DT_type, parent) -> Tree.DTree:

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
           if IG > highest:
            highest = IG
            highest_att = attribute

        elif DT_type == "gini":
            IG = IG_gini(attribute, labels, df, target_count)
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
        # print("Going down branch " + label)
        # print(sub_df)
        lower_branch.append(build_DT(attributes, sub_df, DT_type, label))

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

    return
        
     

# entropy_Tree = build_DT(attributes, df, "entropy", "root")
# print_tree(entropy_Tree, 1)

gini_Tree = build_DT(attributes, df, "gini", "root")
print_tree(gini_Tree, 1)