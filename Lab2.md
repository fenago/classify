## Part 2: Building a Decision Tree Model
In this part, we will continue working with the Online Shoppers Intention dataset from the previous lab. We will build a Decision Tree model to predict whether a user will make a purchase on the online site.

### Step 1: Train a Decision Tree model
```python
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)

```
Train a Decision Tree model using the training data. Decision Trees are useful for classification problems as they can handle both categorical and numerical features.
### Step 2: Get model hyperparameters
```python
dt.get_params()

```
Check the hyperparameters of the trained Decision Tree model. Hyperparameters are parameters that are not learned by the model but are set by the user.
### Step 3: Verify feature names
```python
type(X_train)
type(dv.get_feature_names_out())
type(dt.feature_importances_)
dv.get_feature_names_out()

```
Verify the feature names in the trained model. Understanding the feature names is essential when interpreting the model's results.
### Step 4: Get model properties
```python
def get_properties(model):   
  return [i for i in model.__dict__ if i.endswith('_')] 
get_properties(dt)

```
Get the properties of the trained model. These properties can provide useful insights into the model's performance.
### Step 5: Check feature importances
```python
feature_names = dv.feature_names_
for i, j in zip(feature_names, dt.feature_importances_): 
    print('%.3f' % j, i)

```
Check the importance of each feature in the trained model. Feature importances can help you understand which features are the most important for making predictions.

### Step 6: Evaluate model accuracy
```python
from sklearn.metrics import accuracy_score, classification_report
print('Accuracy: %.3f' % accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
df['revenue'].value_counts()

```
Evaluate the accuracy of the trained model. You can also use other metrics like precision, recall, or F1-score, depending on the balance of your target variable.
### Step 7: Understand the confusion matrix
A confusion matrix helps you understand the performance of your classification model by showing the number of true positives, true negatives, false positives, and false negatives.
```python
from sklearn.metrics import confusion_matrix
cnf_matrix = confusion_matrix(y_test, y_pred)
print(cnf_matrix)

```
### Step 8: Inspect predictions
```python
pred_y = dt.predict(X_test)
print("The first 10 predictions {}".format(pred_y[:10].round(0)))
print("The real first 10 labels {}".format(y_test[:10]))

```
Inspect the first 10 predictions made by the model and compare them to the actual labels. This can help you understand the performance of your model on individual instances.

### Step 9: Make predictions on new values
```python
def model_prediction(item, dv, model):
    X = dv.transform([item])
    y_pred = model.predict(X)
    return y_pred[0]

item = df_train_full.iloc[[213]].to_dict('records')[0]
model_prediction(item, dv, dt)

```
Use the trained model to make predictions on new data. This is useful when you want to apply the model to real-world situations.
### Step 10: Convert a row to a dictionary
```python
df_train_full.iloc[[213]].to_dict('records')[0]

```
Convert any row in the dataset to a dictionary. This is important when making predictions, as the input data must be in the correct format.
### Step 11: Create a DataFrame for new values
```python
myItem = {'administrative': [6],
 'administrative_duration': [94.6],
 'informational': [0],
 'informational_duration': [2.0],
 'productrelated': [15],
 'productrelated_duration': [1933.559259],
 'bouncerates': [0.005333333],
 'exitrates': [0.026377261],
 'pagevalues': [167.806338478],
 'specialday': [0.0],
 'month': ['nov'],
 'operatingsystems': [2],
 'browser': [2],
 'region': [4],
 'traffictype': [2],
 'visitortype': ['returning_visitor'],
 'weekend': [False]}

newDF = pd.DataFrame.from_dict(myItem)

item = newDF.to_dict('records')[0]

model_prediction(item, dv, dt)

```
Create a DataFrame for new values and use it to make predictions with the trained model. This allows you to make predictions on real-world data and assess the performance of your model in practice.

### In conclusion:
in this lab, we have built a Decision Tree model using the Online Shoppers Intention dataset, evaluated its performance, and made predictions on new data. Now, try to apply the same steps to a new dataset, the UCI Credit Card Default dataset, available at:

https://raw.githubusercontent.com/fenago/datasets/main/UCI_Credit_Card.csv

Create a new Google Colab notebook and load the data into a DataFrame named df. Use the skills you have learned in this lab to explore, preprocess, and build a model using the new dataset. Good luck!
