import pandas as pd
import numpy as np
import Tree
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import scipy

# Input Values: change to correct parameters 

# run_type can be: 1)"entropy" 2)"gini" 3)"ME" 4)"bag_forest"
run_type = "gini" 
training_path = 'Project1:RandomForests/data/Training.csv'
testing_path = 'Project1:RandomForests/data/agaricus-lepiota - testing.csv'
use_testing = False
target_column = 'class'
not_an_attribute = ['id', 'class']



# Training data set
df_train = pd.read_csv(training_path)

# Sectioning data into a training subset and a validation subset
df_train, df_validation = train_test_split(df_train, test_size=0.2)
df_validation = df_validation.sample(frac=1).reset_index(drop=True)

# Testing data set (No target values provided)
df_testing = pd.read_csv(testing_path) 

# Removing column names that aren't attributes from list of columns
attributes = list(df_train.columns.values)
for item in not_an_attribute:
    attributes.remove(item)
# print(attributes)

# Setting binary target column of 1's and 0's from provided target column 
target = LabelEncoder()
df_train['target'] = target.fit_transform(df_train[target_column])
df_validation['val_target'] = target.fit_transform(df_validation[target_column])
validating = df_validation[['val_target']]
# print(df_train.head())

# Getting list of target values from validation set for accuracy checking
validation = []
for i in range(len(validating)):
    validation.append(validating.iloc[i, 0])



# 
# Build binary value list gathers column counts for the target and 
# fills in empty counts when value doesn't exist in DataFrame
# 
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


# 
# Returns entropy value
# 
def entropy(a: list): 
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


# 
# Caculates IG using entropy
# 
def IG_entropy(attribute, labels: list, df: pd.DataFrame, tc: list):
    IG = entropy(tc)
    total = df.shape[0]
    for label in labels:
        babyFrame = df.loc[df[attribute] == label]
        babytotal = babyFrame.shape[0]
        ltc = babyFrame['target'].value_counts()
        count = build_binary_value_list(ltc)
        IG = IG - ((babytotal/total)*entropy(count))
    return IG


# 
# Returns gini value
# 
def gini(a):
    p = 0
    a_sum = sum(a)
    for i in range(len(a)):
        p += np.square(a[i] / a_sum)
    return 1 - p


# 
# Calculates IG using gini
# 
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


# 
# Returns ME value
# 
def ME(a):
    a_sum = sum(a)
    return 1-max(a[0]/a_sum, a[1]/a_sum)


# 
# Calculates IG using ME
# 
def IG_ME(attribute, labels: list, df, tc): 
    IG = ME(tc)
    total = df.shape[0]
    for label in labels:
        babyFrame = df.loc[df[attribute] == label]
        babytotal = babyFrame.shape[0]
        ltc = babyFrame['target'].value_counts()
        count = build_binary_value_list(ltc)
        IG = IG - ((babytotal/total)*ME(count))
    return IG


# 
# Returns True if Chi 2 if information gained 
# reaches the desired threshold
# 
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


# 
# Returns Chi 2 boolean from chi function 
# 
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


# 
# Returns a DT using desired method. 
# DT is built recursively.
# Note: chi 2 is not currently being used to 
# determine useful attributes although it can 
# easily be uncommented
# 
def build_binary_DT(attributes: list, df, DT_type, parent) -> Tree.DTree:

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
                # if chi2:
                highest = IG
                highest_att = attribute

        elif DT_type == "gini":
            IG = IG_gini(attribute, labels, df, target_count)
            chi2 = chi_info_finder(attribute, df, labels)
            if IG > highest:
                # if chi2:
                highest = IG
                highest_att = attribute

        elif DT_type == "ME": 
            IG = IG_ME(attribute, labels, df, target_count)
            chi2 = chi_info_finder(attribute, df, labels)
            if IG > highest:
                # if chi2:
                highest = IG
                highest_att = attribute
            
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


# 
# Prints DT in a readable/understandable format
# 
def print_tree(tree: Tree.DTree, level:int):

    print("Tree Level: " + str(level))
    print(tree.node) 

    for branch in tree.attributes:
        print_tree(branch, level+1)


# 
# Used to go down branches an reach a prediction
# 
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


# 
# Returns list of predicted values
# 
def predict(tree: Tree.DTree, testing)-> list: 
    predictions = []
    for i in range(len(testing)):
        # print("Showing attributes for row " + str(i) + "**********************")
        # print(testing.iloc[i])
        predictions.append(traverse_tree(tree, testing.iloc[i]))
    # print(predictions)
    return predictions


# 
# Returns accuracy when comparing two lists of values, 
# first value is true value set
# 
def accuracy(validation, predictions):
    return accuracy_score(validation, predictions)


# 
# Builds a list of DT that are used together as a forest,
# method used by IG can be selected as well as the number
# trees to make and the size of the training set they use
# 
def plant_forest(attributes: list, df, DT_type: str, forest_size: int, sample_size: int):
    forest = []
    at = []
    for i in range(forest_size):
        sub_df = df.sample(n=sample_size)
        at.extend(attributes)
        forest.append(build_binary_DT(at, sub_df, DT_type, "root"))
    return forest


# 
# Returns a list of predicted values from all trees in the forest
# 
def get_consensous(all_predictions:pd.DataFrame) -> list:
    final_prediction = []
    for i in range(all_predictions.shape[1]):
        final_prediction.append(all_predictions[i].max())
    return final_prediction


# 
# If statement set for running the desired type of tree base on 
# configuration at the top of the file
# 
if run_type == "entropy":
    entropy_Tree = build_binary_DT(attributes, df_train, "entropy", "root")
    # print_tree(entropy_Tree, 1)
    prediction = predict(entropy_Tree, df_validation)
    print("Entropy accuracy = ", str(accuracy(validation, prediction)))

    if use_testing:
        test_prediction = predict(entropy_Tree, df_testing)
        # print(test_prediction)

        solutions = pd.DataFrame(data = df_testing['id'])
        solutions['class'] = test_prediction
        solutions['class'] = solutions['class'].replace(1,'p')
        solutions['class'] = solutions['class'].replace(0, 'e')
        # print(solutions)

        solutions.to_csv('Project1:RandomForests/data/solutions_entropy.csv', index=False)

elif run_type == "gini":
    gini_Tree = build_binary_DT(attributes, df_train, "gini", "root")
    # print_tree(gini_Tree, 1)
    prediction = predict(gini_Tree, df_validation)
    print("Gini accuracy = ", str(accuracy(validation, prediction)))

    if use_testing:
        test_prediction = predict(gini_Tree, df_testing)
        # print(test_prediction)

        solutions = pd.DataFrame(data = df_testing['id'])
        solutions['class'] = test_prediction
        solutions['class'] = solutions['class'].replace(1,'p')
        solutions['class'] = solutions['class'].replace(0, 'e')
        # print(solutions)

        solutions.to_csv('Project1:RandomForests/data/solutions_gini.csv', index=False)

elif run_type == "ME":
    ME_Tree = build_binary_DT(attributes, df_train, "ME", "root")
    # print_tree(ME_Tree, 1)
    prediction = predict(ME_Tree, df_validation)
    print("ME accuracy = ", str(accuracy(validation, prediction)))

    if use_testing:
        test_prediction = predict(ME_Tree, df_testing)
        # print(test_prediction)

        solutions = pd.DataFrame(data = df_testing['id'])
        solutions['class'] = test_prediction
        solutions['class'] = solutions['class'].replace(1,'p')
        solutions['class'] = solutions['class'].replace(0, 'e')
        # print(solutions)

        solutions.to_csv('Project1:RandomForests/data/solutions_ME.csv', index=False)

elif run_type == "bag_forest":
    # (attributes, training set, method "gini" "entropy" or "ME", # of trees, # of samples)
    forest = plant_forest(attributes, df_train, "gini", 30, 500)
    forest_predictions = []

    for tree in forest:
        # print(tree.node)
        # print(len(tree.attributes))
        print_tree(tree,1)
        forest_predictions.append(predict(tree, df_validation))

    for i in forest_predictions:
        flag = False
        for j in i:
            if j == None:
                print('None')
                flag = True
                break
        if not flag:
            print(accuracy(validation, i))

    FP_array = np.array(forest_predictions)
    df_predictions = pd.DataFrame(data=FP_array)
    forest_prediction = get_consensous(df_predictions)

    print(accuracy(validation, forest_prediction))
    
    # print(forest_prediction) 

    