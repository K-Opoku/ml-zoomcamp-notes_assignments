#  ML Zoomcamp – Module 2 Notes  

*Car Price Prediction Project*

---

## 🔹 Overview
Module 2 walks through building a **linear regression model** to predict car prices.  
It covers the full workflow:
- Data preparation
- Exploratory data analysis (EDA)
- Validation frameworks
- Linear regression (theory + implementation)
- Feature engineering (numerical + categorical)
- Model evaluation with RMSE
- Regularization & tuning
- Using the trained model for predictions

---




---

## 2.1 – Data Preparation

The data preparation stage is where the dataset is loaded using the NumPy and Pandas libraries.  
At this stage, I explored the structure of the data and its features, identified missing values, and documented early observations using functions like `.head()`, `.info()`, and `.isnull().sum()`.

Key steps included:
- Loading the dataset with `pd.read_csv()`
- Checking for missing values and data types
- Resetting indices after cleaning
- Preparing the target variable (`price`)
- Applying log transformation to reduce skewness

**Code example:**
```python
import pandas as pd
import numpy as np
df = pd.read_csv("data.csv")
print(df.head())
print(df.info())
df["price"] = np.log1p(df["price"])
```

---

## 2.2 – Exploratory Data Analysis (EDA)

Exploratory Data Analysis (EDA) is the process of understanding the dataset before modeling. In Module 2, I used EDA to explore the car price dataset and identify useful patterns.

I visualized the target (price) using matplotlib and seaborn to assess its distribution. The histogram revealed a right-skewed pattern, which was corrected using log transformation.

**Why EDA Matters:**
- Detects missing values and outliers
- Reveals distributions of numerical features
- Shows relationships between features and the target
- Guides feature engineering decisions

**Code example:**
```python
import matplotlib.pyplot as plt
import seaborn as sns

sns.histplot(df["price"], bins=50)
plt.show()
```

---

## 2.3 – Validation Framework

To ensure fair model evaluation, the dataset was split into three parts:

- **Training set (60%)** – used to fit the model
- **Validation set (20%)** – used to tune and evaluate
- **Test set (20%)** – used only once for final performance check



After splitting, I performed:
- Missing value imputation
- Log transformation of the target
- Feature selection for baseline modeling

---

## 2.4 – Linear Regression

Linear regression is a foundational algorithm that models the target as a weighted sum of input features:

\[ y = w_0 + w_1x_1 + w_2x_2 + ... + w_nx_n \]

**Where:**
- `w0` is the bias (intercept)
- `w1...wn` are the weights (coefficients)
- `x1...xn` are the input features

**Vector Form:**  
\[ y = w_0 + X ⋅ w \]

This compact form allows efficient implementation using matrix operations.

**Training the Model:**
We used the Normal Equation to compute weights:

```python
def train_linear_regression(X, y):
    X = np.column_stack([np.ones(X.shape[0]), X])
    w = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
    return w[0], w[1:]
```

The model was trained on the training set and evaluated on the validation set using Root Mean Squared Error (RMSE):

```python
def rmse(y, y_pred):
    return np.sqrt(((y - y_pred) ** 2).mean())
```

---

## 2.5 – Baseline Modeling

The first model used only numerical features:
- engine_hp
- engine_cylinders
- highway_mpg
- city_mpg
- popularity

This gave a baseline RMSE, which was later improved through feature engineering and regularization.

---

## 2.6 – Feature Engineering

Feature engineering improves model performance by transforming raw data into meaningful inputs.

**Numerical Features:**
- Created `age = 2020 - year` to capture vehicle age
- Applied log transformation to price

**Categorical Features:**
- Selected top 5 most frequent values for each categorical column
- Manually encoded them using one-hot encoding:

```python
for c, values in categories.items():
    for v in values:
        df[f'{c}_{v}'] = (df[c] == v).astype(int)
        features.append(f'{c}_{v}')
```

This allowed the model to learn from brand, transmission type, vehicle size, and other categorical traits.

---

## 2.7 – Regularization

Regularization prevents overfitting by penalizing large weights in the model. It improves generalization and stability.

**L2 Regularization (Ridge Regression):**
Adds a penalty to the loss function:

\[ Loss = ∑(y_i − ŷ_i)^2 + λ∑w_j^2 \]

**Implementation:**
```python
def train_linear_regression_reg(X, y, r=0.01):
    X = np.column_stack([np.ones(X.shape[0]), X])
    XTX = X.T.dot(X)
    XTX += r * np.eye(XTX.shape[0])
    w = np.linalg.inv(XTX).dot(X.T).dot(y)
    return w[0], w[1:]
```

By tuning `r`, we controlled the strength of regularization and reduced overfitting.

---

## 2.8 – Final Model and Deployment

After tuning and feature engineering:
- Retrained the model on train + validation
- Evaluated final performance on test set
- Saved weights for future use
- Used the model to predict prices for new car data

**Key Takeaways:**
- Start with a simple baseline, then iterate
- Use EDA to guide feature selection and transformation
- Split data properly to avoid overfitting
- Encode categorical variables carefully
- Regularization improves generalization
- Final model should be trained on full data and tested once

