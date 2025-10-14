# 🧑‍💻 Machine Learning Zoomcamp – Module 3: Classification
This module introduces classification problems, where the goal is to predict categories (e.g., churn vs. no churn, spam vs. not spam) instead of continuous values (like in regression). We’ll use a customer churn prediction project as the running example.

##  3.1 Churn Prediction Project
Problem setup: Predict whether a customer will churn (leave the service).

Type of ML task: Binary classification (two possible outcomes: churn = 1, no churn = 0).



##  3.2 Data Preparation
The data preparation is very similar to regression. We clean  column names, check missing values, and ensure correct data types to understand the dataset better.

Target variable: churn (1 = churned, 0 = stayed).

Features:

Numerical: tenure, monthly charges, total charges.

Categorical: gender, contract type, payment method.

python
df.columns = df.columns.str.lower().str.replace(' ', '_')
df['churn'] = (df.churn == 'yes').astype(int)
## 3.3 Validation Framework

At validation framework we split data into train/validation/test sets in a ratio of 60-20-20 just like it was done in regression. In this module is when i discovered how to perform the data split using scikit learn because in the previous module(Regression) we focused on how to perform the manually for better and deep understanding of the whole process.
**Note** After splittiv and getting the targets for the split, the target should be deleted from the dataframe split. This should be ensured to prevent leakage.



python
from sklearn.model_selection import train_test_split

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)
## 3.4 Exploratory Data Analysis (EDA)
Churn rate: % of customers who churned.

Feature distributions: Look at histograms and churn rates across categories.

Insights:

Month‑to‑month contracts → higher churn.

Longer tenure → lower churn.

Electronic check → higher churn.

## 3.5 Feature Importance: Churn Rate & Risk Ratio
Churn rate: % of churners in a group.

Risk ratio: Group churn rate ÷ overall churn rate.
**This helps identify high‑risk groups.**

## 3.6 Feature Importance: Mutual Information
Measures how much knowing a feature reduces uncertainty about the target.

Works for categorical features.

Range: 0 (no relation) → higher values (stronger relation).

python
from sklearn.metrics import mutual_info_score

mutual_info_score(df_train.contract, df_train.churn)
## 3.7 Feature Importance: Correlation
There are two uses of correlation in classification:

(a) Correlation with the Target
Treat churn as 0/1 and compute Pearson correlation with numeric features.

Example: tenure is negatively correlated with churn.

python
df_train[['tenure', 'monthlycharges', 'churn']].corr()
(b) Correlation Among Features
Compute correlation matrix across all numeric features.

Detects multicollinearity (redundant features).

Example: total_charges is highly correlated with tenure × monthly_charges.

python
import seaborn as sns
import matplotlib.pyplot as plt

corr_matrix = df_train.corr()
plt.figure(figsize=(8,6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm')
plt.show()
**Summary:**

Correlation with target → tells us which features are predictive.

Correlation among features → tells us which features are redundant.

## 3.8 One‑Hot Encoding (OHE)
Convert categorical variables into binary columns. Thus is done so that the model can understand and train on our data.

Example: contract = month-to-month → [1,0,0].

python
from sklearn.feature_extraction import DictVectorizer

dv = DictVectorizer(sparse=False)
train_dicts = df_train[categorical + numerical].to_dict(orient='records')
X_train = dv.fit_transform(train_dicts)
## 3.9 Logistic Regression
Predicts probabilities of churn.

Uses the sigmoid function to map log‑odds to [0,1].

python
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
model.fit(X_train, y_train)
## 3.10 Model Interpretation
Coefficients: Positive → increases churn probability, Negative → decreases churn probability.

Logistic regression is interpretable and business‑friendly.

## 3.11 Using the Model
Predict churn probability for new customers.

Adjust decision threshold (default = 0.5) depending on business needs.

python
y_pred = model.predict_proba(X_val)[:,1]
## 3.12 Summary
Classification predicts categories, not continuous values.

We explored:

Churn prediction setup

Validation framework

EDA & feature importance

Correlation (with target & among features)

One‑hot encoding

Logistic regression

Model interpretation & usage

## Key Takeaways
Classification is about probabilities, not just labels.

Logistic regression is a baseline model: simple, interpretable, effective.

Correlation has two roles: feature importance (with target) and redundancy detection (among features).

Always validate models with proper splits.
