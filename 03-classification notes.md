# Module 3: Classification – Churn Prediction Teaching Notebook

This is a note i made after completing the classification section of the ml zoomcamp

---

## 1. Introduction & Setup

Classification is about predicting discrete labels (e.g. churn vs non-churn).  
In this section we :

- Loaded and cleaned data  
- Explored features  
- Computed feature importance  
- Encoded categorical variables  
- Trained logistic regression  
- Interpreted model results  



```python
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import mutual_info_classif
2. Load & Prepare Data
python
Copy code
# Download / load the churn dataset
!wget https://…/churn.csv
df = pd.read_csv("churn.csv")

# Normalize column names
df.columns = df.columns.str.lower().str.replace(" ", "_")

# Convert target to binary
df["churn"] = (df["churn"] == "yes").astype(int)

# Handle missing values if needed
# For example:
# df["some_col"] = pd.to_numeric(df["some_col"], errors="coerce")
# df = df.fillna(0)

df.head().T
We checked:

df.dtypes

df.isna().sum()

Value counts of churn

3. Validation Framework
We split the data into train / validation / test using  60-20-20 ratio.

python

df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)
# → train = 60%, val = 20%, test = 20%

y_train = df_train["churn"].values
y_val = df_val["churn"].values
y_test = df_test["churn"].values

drop_cols = ["churn", "customer_id"]
X_train = df_train.drop(columns=drop_cols)
X_val = df_val.drop(columns=drop_cols)
X_test = df_test.drop(columns=drop_cols)
4. Exploratory Data Analysis (EDA)
We explored our features and target:

df_train.describe()

Churn distribution: df_train["churn"].value_counts(normalize=True)

Categorical feature counts: df_train["some_cat"].value_counts()

Crosstabs:

python
pd.crosstab(df_train["some_cat"], df_train["churn"])
For numeric features: histograms, boxplots, scatter plots

Our insight goal was to identify feature levels or numeric ranges that differed for churners vs non-churners.

5. Feature Importance
5.1 Churn Rate & Risk Ratio
python
Copy code
churn_rate = df_train.groupby("cat")["churn"].mean().sort_values()
overall = df_train["churn"].mean()
risk_ratio = churn_rate / overall
Churn rate per level

Risk ratio: relative risk compared to overall churn

5.2 Mutual Information
python
Copy code
def mutual_info_series(series, y):
    le = LabelEncoder()
    s = le.fit_transform(series)
    return mutual_info_classif(s.reshape(-1, 1), y, discrete_features=True)[0]

for c in categorical_cols:
    print(c, mutual_info_series(df_train[c], y_train))
A higher mutual information score implied stronger dependence with the target.

5.3 Correlation
For numeric features:

python
Copy code
corr = df_train[num_cols + ["churn"]].corr()["churn"].sort_values(ascending=False)
print(corr)
This gave us a sense of linear association (bearing in mind limitations with a binary target).

6. Encoding Categorical Variables (One-Hot using DictVectorizer)
python
Copy code
def prepare_dataset(df, categorical, numerical, dv=None, fit_dv=False):
    dicts = df[categorical + numerical].to_dict(orient="records")
    if fit_dv:
        dv = DictVectorizer(sparse=False)
        X = dv.fit_transform(dicts)
    else:
        X = dv.transform(dicts)
    return X, dv

# Suppose chosen features:
categorical = ["some_cat", "another_cat"]
numerical = ["num1", "num2"]

X_train, dv = prepare_dataset(df_train, categorical, numerical, dv=None, fit_dv=True)
X_val, _ = prepare_dataset(df_val, categorical, numerical, dv=dv, fit_dv=False)

# Optionally inspect:
feature_names = dv.get_feature_names_out()
print(feature_names)
7. Training Logistic Regression
python
Copy code
model = LogisticRegression(solver="liblinear", C=1.0, random_state=1)
model.fit(X_train, y_train)

y_pred = model.predict(X_val)
y_pred_proba = model.predict_proba(X_val)[:, 1]
We inspected coefficients:

python
Copy code
coef = model.coef_[0]
sorted_idx = coef.argsort()
for idx in sorted_idx:
    print(feature_names[idx], coef[idx])
Positive coefficient → increased churn probability

Negative → decreased churn probability

8. Retrain on Full Data & Predict on Test
python
Copy code
X_full_train, dv = prepare_dataset(
    pd.concat([df_train, df_val]),
    categorical,
    numerical,
    dv=None,
    fit_dv=True
)
y_full_train = pd.concat([df_train["churn"], df_val["churn"]]).values

model = LogisticRegression(solver="liblinear", C=1.0, random_state=1)
model.fit(X_full_train, y_full_train)

X_test_enc, _ = prepare_dataset(df_test, categorical, numerical, dv=dv, fit_dv=False)
y_test_pred = model.predict(X_test_enc)
y_test_proba = model.predict_proba(X_test_enc)[:, 1]
Then our pipeline was ready for evaluation in the next module.
