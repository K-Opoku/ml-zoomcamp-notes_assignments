# 🧠 ML Zoomcamp – 01_intro Notes (Conceptual Summary)

## 1. What is Machine Learning?
Machine learning is the process of teaching computers to recognize patterns from data rather than programming explicit rules.  
Unlike traditional software, ML systems learn from examples and improve over time.  
This shift is crucial in domains like telecom, where customer behavior is too complex to hard-code.

---

## 2. ML vs Rule-Based Systems
Rule-based systems rely on human-defined logic (e.g., “if age > 60, then high risk”), but ML systems learn these patterns automatically from data and predict the target(unknown object) based on it's features.  
ML can handle more data and adjust to changes easily, even when the rules aren’t clear. This is the reason why ML is better and efficient than Rule Based Systems.

---

## 3. Supervised Learning
This is the most common ML type, where the model learns from labeled data.  In this learning the model is taught by examples identifying patterns to be able to predict the target of an unknown target.
**Types of surpervised learning:**
- Regression:        - numerical value
- Classification     - and others ...

---

## 4. CRISP-DM Framework
A structured approach to ML projects. This is the methodology and foundation for organising ML projects
- **Business Understanding**: What problem are we solving? The **Business understanding** is understood by identifying the problem you are solving, and how it can be solved (if it can be solved with ML then you move on to the next steps).
   (**Tip: At this stage set a measurable goal to be able to track success at the end of the project**)
- **Data Understanding**: What data do we have? This step is where we gather the data needed for the problem solving. The larger the set of data the better the trainning of the model. Remove all inconsistent,noise and data with errors to maintaon accuracy, because if the data collected is poor the model will learn the wrong pattern. After this, check if the remaining data is enough. If not, find additional data.
- **Data Preparation**: This stage is where the data is cleaned and transformed for the model to learn. Before training a machine learning model, we need to prepare the data so it can be understood by the algorithm. This process includes selecting useful features and converting them into numbers.

Let’s say we’re trying to detect spam emails. We choose features like:

- Does the email contain the word “prize”?
- Does it include the word “click”?
- Does it have a link?
- Is the message very short?

Each email is then represented as a row of 0s and 1s, showing which features are present:

- `1` means the feature is present
- `0` means it’s not

Even if only one feature is active (e.g., just the word “prize”), and the email is spam, the model can still learn that this single clue is important.

The final column is the label:
- `1` = spam
- `0` = not spam

This structured format allows the model to learn patterns from past examples and make predictions on new emails.

> Machine learning doesn’t need all features to be active—it learns from patterns across many examples.
 
EXAMPLE:  **📊 Example Table – Data Preparation for Spam Detection**

| Email_ID | Contains “prize” | Contains “click” | Has link | Is short | Spam |
|----------|------------------|------------------|----------|----------|------|
| A        | 1                | 1                | 1        | 0        | 1    |
| B        | 0                | 0                | 0        | 1        | 0    |
| C        | 1                | 0                | 0        | 0        | 1    |
| D        | 0                | 1                | 1        | 1        | 0    |

- `1` means the feature is present in the email  
- `0` means the feature is not present  
- The last column (`Spam`) is the label: `1` = spam, `0` = not spam

> Even if only one feature is present (like in Email C), the model can still learn that this pattern often signals spam.

- **Modeling**: .This stage is the choosing and training of models and eventually pick the best one
- **Evaluation**: This stage is where the performance of the model is been measured and compared to the goal set earlier in the bussiness understanding stage.
- **Deployment**: Making the model usable in production


---

## 5. Model Selection Process
Key steps:
- **Train/Validation/Test Split**:  
In machine learning, the dataset is often split into three parts:

1. **Training set** – used to train different models.
2. **Validation set** – used to compare model performance and select the best one.
3. **Test set** – used to evaluate the final selected model on completely unseen data.

After selecting the best model using the training and validation sets, we often retrain it using both sets combined. Then we test it on the test set to check how well it performs in real-world scenarios. This helps ensure the model is not just good on known data, but also generalizes well to new data.


---

## 6. Environment Setup
Installed Python, Jupyter, Pandas, NumPy, and other ML tools using Anaconda.  
This ensures reproducibility and clean package management.

---

## 7. NumPy & Linear Algebra
This nodule incluses numpy and linear algebra as well as how they are been used.

---

## 8. Pandas Basics
Explored:
- DataFrames and Series
- Missing value handling
- Aggregation and filtering

These are essential for real-world data wrangling.

---

## ✍️ Personal Reflection
This module clarified the difference between coding and learning.  
I now see ML as a strategic tool—not just a technical skill.  

