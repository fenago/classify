# Lab: Analyzing Credit Card Defaulters
### Step 1: Overview
In this lab, you will analyze the characteristics of customers most likely to default on their credit card payments using univariate and bivariate analysis techniques. By the end of this lab, you will be able to build a profile of a customer who is the most statistically likely to default on their credit card payments.

### Step 2: Introduction
In a previous lab, we analyzed online shoppers' purchasing intent. Now, we will analyze credit card payments of customers and build a profile of those most likely to default. This profile can be used by banks or lending facilities to detect potential defaulters and take appropriate actions in a timely manner.

### Step 3: Import the Data
Import the required libraries:
```python
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline

```
Read/import the dataset into the work environment:
```python
df = pd.read_excel('default_credit.xls')
df.head(5)

```
Check the metadata of the DataFrame:
```python
df.info()

```
Check the descriptive statistics for the numerical columns in the DataFrame:
```python
df.describe().T

```
Check for null values:
```python
df.isnull().sum()

```
### Step 4: Data Preprocessing
Clean the data of any errors, identify unique values, and make the data more meaningful by forming groups. Ensure data consistency, such as displaying categorical columns as integers.

### Step 5: Data Preprocessing
Check the unique values in various columns to identify subcategories and understand the distribution of data:
```python
print('SEX ' + str(sorted(df['SEX'].unique())))
print('EDUCATION ' + str(sorted(df['EDUCATION'].unique())))
print('MARRIAGE ' + str(sorted(df['MARRIAGE'].unique())))
print('PAY_0 ' + str(sorted(df['PAY_0'].unique())))
print('default.payment.next.month ' + str(sorted(df['default payment next month'].unique())))

```
Club categories in the EDUCATION and MARRIAGE columns to match the data description:
```python
fill = (df.EDUCATION == 0) | (df.EDUCATION == 5) | (df.EDUCATION == 6)
df.loc[fill, 'EDUCATION'] = 4

fill = (df.MARRIAGE == 0)
df.loc[fill, 'MARRIAGE'] = 2

```
Check the unique values in the EDUCATION and MARRIAGE columns after clubbing the values:
```python
print('EDUCATION ' + str(sorted(df['EDUCATION'].unique())))
print('MARRIAGE ' + str(sorted(df['MARRIAGE'].unique())))

```
Rename the PAY_0 column to PAY_1 and the default payment next month column to DEFAULT to maintain consistency with the naming of other columns:
```python
df = df.rename(columns={'default payment next month': 'DEFAULT', 'PAY_0': 'PAY_1'})
df.head()

```
In this step, you have checked the unique values in various columns, combined a few subcategories, and renamed a few columns to make data analysis more efficient and easier to interpret. The next step will involve exploratory data analysis.
## Exploratory Data Analysis (EDA)
In EDA, we investigate data to find hidden patterns and outliers with the help of visualization. EDA can be split into three parts:

Univariate analysis
Bivariate analysis
Correlation
First, let's look at Univariate Analysis. Univariate analysis involves analyzing each feature (column) individually to uncover patterns or distributions. In this case, we'll analyze the categorical columns (DEFAULT, SEX, EDUCATION, and MARRIAGE).

DEFAULT column:
```python
sns.countplot(x="DEFAULT", data=df)
df['DEFAULT'].value_counts()

```
SEX column:
```python
sns.countplot(x="SEX", data=df)
df['SEX'].value_counts()

```
EDUCATION column:
```python
sns.countplot(x="EDUCATION", data=df)
df['EDUCATION'].value_counts()

```
MARRIAGE column:
```python
sns.countplot(x="MARRIAGE", data=df)
df['MARRIAGE'].value_counts()

```
After analyzing the distribution of data in each of these columns, you can draw some inferences about the dataset. For example, around 22% of customers have defaulted on their payments, there are more females than males in the dataset, most customers have either a graduate school or university degree, and the number of single people is higher than the number of married people.

Next, you'll proceed with Bivariate Analysis and Correlation to further explore relationships between variables in the dataset.
Bivariate Analysis

Bivariate analysis is performed between two variables to look at their relationship.

In this section, you will consider the relationship between the DEFAULT column and other columns in the DataFrame with the help of the crosstab function and visualization techniques.

The SEX column versus the DEFAULT column:

In this section, you will look at the relationship between the SEX and DEFAULT columns by plotting a count plot with the hue as DEFAULT to compare the number of male customers who have defaulted with the number of female customers who have defaulted:
```python
sns.set(rc={'figure.figsize':(15,10)})
edu = sns.countplot(x='SEX', hue='DEFAULT', data=df)
edu.set_xticklabels(['Male','Female'])
plt.show()

```
From the preceding graph, you can see that females have defaulted more than males. But this graph doesn't show us the complete picture. To determine what percentage of each sex has defaulted, we will perform cross-tabulation.

Cross-tabulation is a technique used to show the relationship between two or more categorical values. For example, in this scenario, we would like to find the relationship between DEFAULT and SEX. A crosstab table will show you the count of customers for each.
We can also find the percentage distribution for each pair by passing in the normalize='index' parameter, as follows:
```python
pd.crosstab(df.SEX,df.DEFAULT,normalize='index',margins=True)

```
As you can see, around 24% of male customers have defaulted and around 20% of female customers have defaulted.

In the next exercise, we will evaluate the relationship between the EDUCATION, MARRIAGE and DEFAULT columns.
Plot a count plot using seaborn for the EDUCATION and DEFAULT columns, setting the hue as DEFAULT:
```python
sns.set(rc={'figure.figsize':(15,10)})
edu = sns.countplot(x='EDUCATION', hue='DEFAULT', data=df)
edu.set_xticklabels(['Graduate School','University',\
                     'High School','Other'])
plt.show()

```
Observe the count plot for each subcategory. You can conclude from the plot that a greater number of defaults happen for customers whose highest qualification is University, but it is advisable to first perform cross-tabulation to find the exact count.

To determine which subcategory has a higher default percentage, perform cross-tabulation.

Multivariate analysis 
Multivariate Analysis involves analyzing the interactions between multiple variables at once. This helps in understanding the complex dependencies between variables and how they affect the target variable.

In this section, we will explore the relationship between the SEX, EDUCATION, and DEFAULT columns and visualize the interactions between these variables. We will use a catplot from the seaborn library to analyze these relationships.

Visualizing the Relationship between SEX, EDUCATION, and DEFAULT:

To visualize the relationship between SEX, EDUCATION, and DEFAULT columns, we can use the seaborn catplot function as follows:
```python
sns.catplot(x="SEX", y="EDUCATION", hue="DEFAULT", kind="box", data=df)
plt.xticks([0,1], ['Male', 'Female'])
plt.yticks([0,1,2,3], ['Graduate School', 'University', 'High School', 'Other'])
plt.show()

```
From the plot, we can observe that male customers who have graduated from high school are more likely to default on their loans, whereas female customers who have completed graduate school or university have a lower likelihood of defaulting.

Visualizing the Relationship between AGE, EDUCATION, and DEFAULT:

Similarly, we can also explore the relationship between AGE, EDUCATION, and DEFAULT columns by using a scatterplot from the seaborn library:
```python
sns.scatterplot(x="AGE", y="EDUCATION", hue="DEFAULT", data=df)
plt.yticks([0,1,2,3], ['Graduate School', 'University', 'High School', 'Other'])
plt.show()

```
The scatterplot demonstrates that customers who are younger and have a high school education are more likely to default on their loans compared to older customers with graduate school or university degrees. It also shows that the default rate tends to decrease as the customer's age increases.

Visualizing the Relationship between PAY_1, PAY_2, and DEFAULT:

Now, let's analyze the relationship between PAY_1, PAY_2, and DEFAULT columns using a scatterplot from the seaborn library:
```python
sns.scatterplot(x="PAY_1", y="PAY_2", hue="DEFAULT", data=df)
plt.show()

```
From this scatterplot, we can observe that customers who have delayed payments in both the first and second months have a higher likelihood of defaulting on their loans.

In summary, multivariate analysis allows us to explore complex relationships between multiple variables and gain deeper insights into the data. By visualizing these interactions, we can better understand the dependencies between variables and their impact on the target variable. This can help businesses make informed decisions and take necessary actions to mitigate risks associated with loan defaults.
Correlation Analysis
Correlation analysis is a statistical technique used to measure the strength and direction of the relationship between two or more variables. It helps us understand how one variable changes when another variable changes. In the context of this credit card default dataset, we can use correlation analysis to identify the relationships between various attributes and their impact on the default status.

We will use the Pearson correlation coefficient, which ranges from -1 to 1. A value of -1 represents a perfect negative correlation, 0 represents no correlation, and 1 represents a perfect positive correlation.

### Calculating Correlations:

To calculate correlations between the columns in our dataset, we can use the pandas 'corr' function as follows:
```python
correlations = df.corr()

```
Visualizing Correlations:

A heatmap is an effective way to visualize the correlations between the variables in our dataset. We can use the seaborn library to create a heatmap as follows:
```python
sns.heatmap(correlations, annot=True, cmap="coolwarm", vmin=-1, vmax=1)
plt.show()

```
Analyzing the Correlations:

From the heatmap, we can observe some interesting relationships between the variables and the DEFAULT column:

There is a moderate positive correlation between PAY_1, PAY_2, PAY_3, PAY_4, PAY_5, and DEFAULT. This indicates that customers who have delayed payments in the recent months are more likely to default on their loans.
There is a weak negative correlation between LIMIT_BAL and DEFAULT. This suggests that customers with higher credit limits have a lower likelihood of defaulting.
There is a weak negative correlation between EDUCATION and DEFAULT. This indicates that customers with higher education levels have a lower likelihood of defaulting on their loans.
Summary
In this analysis, we explored a credit card default dataset to understand the relationship between various factors and the likelihood of a customer defaulting on their loans. We employed various data visualization and statistical techniques to analyze the data, such as:

Univariate analysis to study the distribution of individual variables
Bivariate analysis to identify relationships between two variables
Multivariate analysis to explore interactions between multiple variables
Correlation analysis to measure the strength and direction of relationships between variables
Our findings suggest that factors such as payment delays, credit limit, education level, sex, and age can impact the likelihood of a customer defaulting on their loans. Financial institutions can use these insights to identify high-risk customers and develop strategies to mitigate loan default risks.
