# Make Sparse

- make_sparse.py

# Naive Bayes

- main.py
- test.p

## Description

The Naive Bayes implementation uses two scripts to run. These scripts need to be
run in order main.py and then test.py. Main first creates and saves a matrix containing the MAP 
values and the MLE values for the training data set proveded. Test then uses the matrix
created by main to find the predicted targets for the provided testing data. Both scripts 
can be run for a singlular Beta value or run in a loop to test multiple Beta values at
once. 

## Usage

### main.py

```python
# Set the following parameters prior to running

train_sparse = sparse.load_npz('Put path to training .npz file here') 

run_loop = True / False # Set to False to run with single beta value True to loop through many 
 
```
After parameters are correctly set simply running main.py will produce and save the needed matrix. Note:
you may also change where the martix file is being saved to, otherwise it will use the defualt.

### test.py

```python

# Set the following parameters prior to running

run_loop = False # Set to False so that it runs test using single Beta True to run loop 

plot_results = False # Set to True to plot results (Must have true target values)

true_targets = False # Set to True if the test data set has the true target values in the last column

# If run_loop is set to False
 
test_sparse = sparse.load_npz('Path to testing .npz file here')

lh_np = loadtxt('Path to saved MAP and MLE matrix .csv file', delimiter=',')

# If run_loop is set to True

test_sparse = sparse.load_npz('Path to testing .npz file here')

beta_MAP_MLE = loadtxt(' Path to folder containing set of matrices +  /beta_'+str(i)+'.csv', delimiter=',')

 
```
After parameters are correctly set simply running test.py will produce and save a file of the results and sgow plots if set to do so. Note: you may also change where the results file is being saved to, otherwise it will use the defualt. There are also settings to save a file meant for the kaggle competition however these are currently commented out.

# Logistice Regression

- logistic_regression.py


# Questions 6 & 7

- q6_and_q7.py
- q6_and_q7_meth2.py

## Description

There are two methods for implementing the solution for question 6. The first method mentioned is implemented 
by q6_and_q7.py. This method works to get the information gain for each word however it would potentially take days to run and therefore was bypassed for method two. The second method described in the answer to question 6 is implemented by q6_and_q7_meth2.py. It is q6_and_q7_meth2.py that was used to answer question 7. 

## Usage

### q6_and_q7_meth2.py

```python
# Set the following parameters prior to running

train_sparse = sparse.load_npz('Put path to training .npz file here') 

occurrence = int # Set to the desired minimum word usage  
 
```
After parameters are correctly set simply running q6_and_q7_meth2.py will produce and save a file of the words with the highest scores as well as print out the list in the terminal. Note: You may also change the path for the saved file in this script as well.