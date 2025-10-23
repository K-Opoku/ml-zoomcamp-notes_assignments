# 📘 Module 4: Evaluation Metrics for Binary Classification

## 🧠 Prerequisite: What You Learned in Module 3

Before diving into evaluation, let’s recall what Module 3 taught us:
- How to split data into training, validation, and test sets  
- How to train a logistic regression model  
- How to preprocess features and scale them  
- How to generate predictions using `.predict_proba()` and `.predict()`

Now that we can train models and make predictions, we need to **measure how good those predictions are**. That’s what Module 4 is all about.

---

## 🎯 Why Evaluation Matters

Evaluation helps us answer:
- Is my model better than random guessing?  
- What kinds of mistakes does it make?  
- Can I trust its confidence scores?  
- What threshold should I use to make decisions?  
- Will it perform well on new data?  

We use **metrics** to answer these questions. Metrics turn predictions into insight.

---

## ✅ Accuracy: The Starting Point

**Definition**: Accuracy is the percentage of correct predictions.

\[
\text{Accuracy} = \frac{\text{Correct Predictions}}{\text{Total Predictions}}
\]

**Example**: If your model predicts correctly 80 out of 100 times, accuracy = 80%.

**Warning**: Accuracy can be misleading when classes are imbalanced.

**Analogy**: Imagine a model that always predicts “converted.” If 62% of leads convert, it gets 62% accuracy — but it’s not learning anything.

---

## 🧪 Dummy Model: Your Baseline

To test whether your model is learning, compare it to a **dummy model** that makes no effort to learn.

```python
from sklearn.dummy import DummyClassifier

dummy = DummyClassifier(strategy='most_frequent')
dummy.fit(X_train, y_train)
print("Dummy accuracy:", dummy.score(X_val, y_val))
```

If your model doesn’t beat the dummy, it’s not useful.

---

## 🧮 Confusion Matrix: The Core of Evaluation

The confusion matrix breaks predictions into four categories:

|                       | Predicted Positive | Predicted Negative |
|-----------------------|--------------------|--------------------|
| **Actual Positive**   | True Positive (TP) | False Negative (FN)|
| **Actual Negative**   | False Positive (FP)| True Negative (TN) |

**Definitions:**
- **True Positive (TP)**: Model correctly predicts a positive case.  
- **False Positive (FP)**: Model wrongly predicts positive for a negative case.  
- **False Negative (FN)**: Model misses a positive case.  
- **True Negative (TN)**: Model correctly predicts a negative case.  

**Code:**

```python
from sklearn.metrics import confusion_matrix

y_pred_binary = y_pred_proba >= 0.5
cm = confusion_matrix(y_val, y_pred_binary)
print(cm)
```

---

## 🎯 Precision and Recall

These two metrics help us understand how good the model is at catching positives.

### 🔹 Precision
**Definition:** Of all predicted positives, how many were correct?

\[
\text{Precision} = \frac{TP}{TP + FP}
\]

**Analogy:** If you say 10 people will convert, and only 6 actually do, your precision is 60%.

### 🔹 Recall
**Definition:** Of all actual positives, how many did we catch?

\[
\text{Recall} = \frac{TP}{TP + FN}
\]

**Analogy:** If 10 people actually convert and you only find 6, your recall is 60%.

### 🔹 F1 Score
**Definition:** Harmonic mean of precision and recall.

\[
F_1 = 2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}
\]

Use F1 when you want a single number that balances both.

---

## ⚖️ Threshold Tuning

Most models output probabilities. To make decisions, we apply a threshold (default = 0.5).

Changing the threshold changes precision and recall:

- Lower threshold → more positives → higher recall, lower precision  
- Higher threshold → fewer positives → higher precision, lower recall  

**Code:**

```python
thresholds = np.arange(0, 1.01, 0.01)
precision_scores = []
recall_scores = []

for t in thresholds:
    y_pred_binary = y_pred_proba >= t
    precision_scores.append(precision_score(y_val, y_pred_binary))
    recall_scores.append(recall_score(y_val, y_pred_binary))
```

**Plot the tradeoff:**

```python
plt.plot(thresholds, precision_scores, label='Precision')
plt.plot(thresholds, recall_scores, label='Recall')
plt.xlabel('Threshold')
plt.ylabel('Score')
plt.legend()
plt.grid()
plt.show()
```

---

## 📈 ROC Curve and AUC

### 🔹 ROC Curve
Plots **True Positive Rate (Recall)** vs **False Positive Rate**

### 🔹 AUC (Area Under Curve)

**Definition:** Measures how well the model ranks positives above negatives.

\[
AUC = P(\text{Positive ranks higher than Negative})
\]

- AUC = 1.0 → perfect ranking  
- AUC = 0.5 → random guessing  

**Code:**

```python
from sklearn.metrics import roc_curve, roc_auc_score

fpr, tpr, _ = roc_curve(y_val, y_pred_proba)
auc = roc_auc_score(y_val, y_pred_proba)
```

**Plot:**

```python
plt.plot(fpr, tpr, label=f'AUC = {auc:.2f}')
plt.plot([0,1],[0,1],'--',label='Random')
plt.xlabel('FPR')
plt.ylabel('TPR')
plt.legend()
plt.grid()
plt.show()
```

---

## 🔁 Cross-Validation

To make sure your metrics are stable, use **K-Fold Cross-Validation**.

**Code:**

```python
from sklearn.model_selection import KFold

kfold = KFold(n_splits=5, shuffle=True, random_state=1)
scores = []

for train_idx, val_idx in kfold.split(df_fulltrain):
    # Split and train
    # Predict and compute AUC
    scores.append(roc_auc_score(y_val, y_pred))
```

**Then report:**

```python
print("Mean AUC:", np.mean(scores))
print("Std Dev:", np.std(scores))
```

---

## 🧠 Summary

- Accuracy is not enough — always compare to a dummy  
- Confusion matrix shows types of errors  
- Precision and recall reveal tradeoffs  
- ROC AUC measures ranking quality  
- Threshold tuning aligns model with business goals  
- Cross-validation ensures generalization  

---

## 🧪 Homework Concepts

- Feature importance via AUC  
- Threshold tuning for precision/recall  
- F1 score optimization  
- 5-fold cross-validation  
- Hyperparameter tuning (C values)  
- Final test set evaluation  
