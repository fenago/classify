## Classifier Lab (Part 1)

In this hands-on lab, you will learn how to analyze and preprocess a dataset using Python and various libraries. You will be able to understand the benefits of each step and visualize the data to gain insights. Finally, you will compare different machine learning algorithms to find the best one for your dataset.

### Step 1: Import your libraries

Before starting any data analysis, you need to import the necessary libraries. This step imports the libraries required for this lab, such as numpy, pandas, matplotlib, seaborn, and scikit-learn.

```python
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import seaborn as sns
import sklearn.metrics as metrics
%matplotlib inline

```

### Step 2: Load your data

Load your dataset into a pandas dataframe. For this example, we will use a placeholder dataset, but you can replace it with your own data.
```python
df = pd.read_csv('./data/<put your data here>')
```

### Step 3: Run these commands 1 cell at a time

These commands will help you explore the dataset and understand its basic characteristics such as the number of rows, summary statistics, shape, data types, unique values, and correlations.
```python
len(df)
df.describe()
df.shape
df.info()
df.nunique()
df.corr()

```

Important:  make sure that you analyze and pull value from EACH of those commands.  Especially the describe and corr() commands.  What is the data telling you?

### Step 4: Basic Data Cleaning

Clean the data by converting column names and string values to lowercase and replacing spaces with underscores. This will make it easier to work with the data.
```python
df.columns = df.columns.str.lower().str.replace(' ', '_')
string_columns = list(df.dtypes[df.dtypes == 'object'].index)
for col in string_columns:
    df[col] = df[col].str.lower().str.replace(' ', '_')
df.head()

```
Important, notice that ALL columns names (features) are now lowercase!

### Step 5: Create Visuals so you can gain a business understanding

Visualize the distribution of your target variable to identify potential data imbalances and gain insights into the dataset.
```python
plt.figure(figsize=(6, 4))
sns.histplot(df.<replace with your target variable>, bins=40, color='black', alpha=1)
plt.ylabel('Frequency')
plt.xlabel('<replace with your target variable>')
plt.title('PUT A LABEL ON IT')
plt.show()

```
What is this telling you?  Is the target balanced?

### Step 6: Check for null values

Make sure there are no null values in the dataset, as they can negatively affect the performance of your machine learning model.
```python
df.isnull().sum()

```

Important, if you notice any null values, then you will have to treat them before the data will run in most machine learning models.

### Step 7: Delete columns

**If necessary**, remove unnecessary columns from the dataset.
```python
# Uncomment and put in the features to drop
# df = df.drop(['x5_latitude', 'x6_longitude', 'x1_transaction_date'], axis=1)
# df.head()

```
### Step 8: Split the data into test, train, and validation sets

Use the train_test_split function from scikit-learn to create an 80/20 split of the data for training and testing.
```python
from sklearn.model_selection import train_test_split
df_train_full, df_test = train_test_split(df, test_size=0.2, random_state=11)
df_train_full = df_train_full.fillna(0)
df_test = df_test.fillna(0)

```
Important:  Notice that we called fillna(0) - although not optimal, this will replace all missing values with a value of "0"

### Step 9: Split the target variable

Separate the target variable from the training and testing data.
```python
y_train = (df_train_full.<replace with your target variable>).values
y_test = (df_test.<replace with your target variable>).values
del df_train_full['<replace with your target variable>']
del df_test['<replace with your target variable>']

```

### Step 10: Convert data frames into a list of dictionaries

Transform the pandas dataframes into lists of dictionaries, where each element in the list is a dictionary representing a record.
```python
dict_train = df_train_full.to_dict(orient='records')
dict_test = df_test.to_dict(orient='records')
```
### Step 11: Convert the list of dictionaries into a feature matrix

Use the DictVectorizer from scikit-learn to convert the list of dictionaries into a feature matrix, which encodes categorical features and creates a matrix representation of the data.
```python
from sklearn.feature_extraction import DictVectorizer
 
dv = DictVectorizer(sparse=False)
 
X_train = dv.fit_transform(dict_train)
X_test = dv.transform(dict_test)
features = dv.get_feature_names_out()
X_test.shape

```
### Step 12: Compare algorithms with the algorithm harness

Compare different machine learning algorithms using cross-validation to identify the best algorithm for your dataset. In this step, we will compare logistic regression, linear discriminant analysis, k-nearest neighbors, decision tree, naive bayes, and support vector machine algorithms.
```python
from sklearn.metrics import roc_auc_score
from time import time
from sklearn.metrics import explained_variance_score,mean_absolute_error,r2_score
from pandas import read_csv
from matplotlib import pyplot
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

models = []
models.append(('LR', LogisticRegression(solver='liblinear')))
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))

results = []
names = []
scoring = 'accuracy'

for name, model in models:
    start = time()
    kfold = KFold(n_splits=10, random_state=7, shuffle=True)
    model.fit(X_train, y_train)
    train_time = time() - start
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
    predict_time = time()-start 
    results.append(cv_results)
    names.append(name)
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)
    print("Score for each of the 10 K-fold tests: ",cv_results)
    print(model)
    print("\tTraining time: %0.3fs" % train_time)
    print("\tPrediction time: %0.3fs" % predict_time)
    print()
    
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

```
This step will give you the relative performance of how the AI will behave in several models.  Important, pick the top 2 algorithms based on:  performance metric (like accuracy) and training time.

### Step 13:  Print out a list of all Classifiers in SKLearn

Print all of the classifiers for context:
```python
from sklearn.utils import all_estimators

def get_classifiers():
    classifiers = []
    for name, ClassifierClass in all_estimators(type_filter='classifier'):
        try:
            clf = ClassifierClass()
            classifiers.append((name, clf))
        except Exception as e:
            pass
    return classifiers

all_classifiers = get_classifiers()
for name, clf in all_classifiers:
    print(name, clf)

```

### Step 14:  Run this data against all models in SKLearn.  (This will take a little time to process)

```python
# Algorithm Harness
# Compare Algorithms with the Algorithm Harness
from sklearn.metrics import roc_auc_score
from time import time
from sklearn.metrics import explained_variance_score, mean_absolute_error, r2_score
from pandas import read_csv
from matplotlib import pyplot
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.utils import all_estimators


def get_classifiers():
    classifiers = []
    for name, ClassifierClass in all_estimators(type_filter='classifier'):
        try:
            clf = ClassifierClass()
            classifiers.append((name, clf))
        except Exception as e:
            pass
    return classifiers


models = get_classifiers()
# evaluate each model in turn
results = []
names = []
scoring = 'accuracy'
for name, model in models:
    try:
        start = time()
        kfold = KFold(n_splits=10, random_state=7, shuffle=True)
        model.fit(X_train, y_train)
        train_time = time() - start
        cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring=scoring)
        predict_time = time() - start
        results.append(cv_results)
        names.append(name)
        msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
        print(msg)
        print("Score for each of the 10 K-fold tests: ", cv_results)
        print(model)
        print("\tTraining time: %0.3fs" % train_time)
        print("\tPrediction time: %0.3fs" % predict_time)
        print()
    except Exception as e:
        print(f"Error with classifier {name}: {e}")

# boxplot algorithm comparison
fig = pyplot.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
pyplot.boxplot(results)
ax.set_xticklabels(names)
pyplot.show()

```

Congratulations!  You have developed several models from scratch!
