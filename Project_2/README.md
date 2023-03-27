# Make Sparse

- make_sparse.py

## Description

Uses scipy library to compress sparse csv data files in a .npz format. For train
data, after compression will split into 20:80 test train files, and then save to
disk. Can also use to create compresed .npz test data for actual predictions.

## Usage

### make_sparse.py

```python

Enter file path to sparse csv file, comment out if compressing sparse csv 
prediction file
# enter train csv file path
train = genfromtxt(
    'file path for training data in sparse csv format', delimiter=',')

Enter file path to save the train and test files, comment out if compressing 
sparse csv prediction file
# enter file path to save the train and test files
sparse.save_npz(
    'file path to save file/csr_train.csv.npz', train_sparse)
sparse.save_npz(
    'file path to save file/csr_test.csv.npz', test_sparse)

#comment out below if compressiong prediction file
split = round(len(train) * 0.80)
indices = np.random.permutation(train.shape[0])
training_idx, test_idx = indices[:split], indices[split:]
train, test = train[training_idx, :], train[test_idx, :]
train_sparse = csr_matrix(train, dtype=np.int16)
test_sparse = csr_matrix(test, dtype=np.int16)

Uncomment to compress sparse csv prediction file 
# prediction = genfromtxt(
#     'file path for test (prediction) data in sparse csv format', 
# delimiter=',')

# prediction_sparse = csr_matrix(prediction, dtype=np.int16)

# sparse.save_npz(
#     'file path to save file/csr_prediction.csv.npz',
#     prediction_sparse)

Enter file path of training data to create train file. Enter file paths for 
saving both the training and testing files. To create test prediction compressed
.npz file, comment out all of the current uncommented lines and uncomment the
current commented lines. Enter file path for both creating and saving the
prediction file.

```
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
After parameters are correctly set simply running test.py will produce and save a file of the results and show plots if set to do so. Note: you may also change where the results file is being saved to, otherwise it will use the defualt. There are also settings to save a file meant for the kaggle competition however these are currently commented out.

# Logistice Regression

- logistic_regression.py

## Description

Runs logistic regression algorithm for text classification.

## Usage

### logistic_regression.py

```python

Enter file path to train_sparse
# load compressed training set
train_sparse = sparse.load_npz(
    'file path to save file/csr_train.csv.npz')

Enter file path to test_sparse
#load compressed testing set
test_sparse = sparse.load_npz(
    'file path to save file/csr_test.csv.npz')

Enter file path to save adjusted weights
# to save current adjusted weights
savetxt('file path to save file/logistic_regression_wts.csv',
        current_weights, delimiter=",")

```

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
