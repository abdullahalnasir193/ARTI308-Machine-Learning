# Decision Trees & Random Forest Lab — ARTI308 Machine Learning

## Overview

This lab covers building and comparing **Decision Tree** and **Random Forest** classifiers on real-world LendingClub loan data (2007–2010). The goal is to predict whether a borrower will fully repay their loan.

---

## Files

| File | Description |
|------|-------------|
| `01-Decision_Trees_and_Random_Forests.ipynb` | Lecture notebook with examples and explanations |
| `02-Decision_Trees_and_Random_Forest_Project.ipynb` | Assignment notebook to complete |
| `loan_data.csv` | Main dataset used for training and testing |
| `kyphosis.csv` | Reference dataset used in the lecture |

---

## Dataset — `loan_data.csv`

9,578 loan records from LendingClub.com with the following columns:

| Column | Description |
|--------|-------------|
| `credit.policy` | 1 if customer meets LendingClub credit criteria, 0 otherwise |
| `purpose` | Purpose of the loan (credit_card, debt_consolidation, etc.) |
| `int.rate` | Interest rate of the loan (e.g. 0.11 = 11%) |
| `installment` | Monthly installment amount owed |
| `log.annual.inc` | Natural log of borrower's annual income |
| `dti` | Debt-to-income ratio |
| `fico` | FICO credit score |
| `days.with.cr.line` | Number of days borrower has had a credit line |
| `revol.bal` | Revolving balance (unpaid at end of billing cycle) |
| `revol.util` | Revolving line utilization rate |
| `inq.last.6mths` | Number of creditor inquiries in last 6 months |
| `delinq.2yrs` | Times 30+ days past due in past 2 years |
| `pub.rec` | Number of derogatory public records |
| `not.fully.paid` | **Target variable** — 1 = not fully paid, 0 = fully paid |

---

## Steps Covered

### 1. Import Libraries
```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
```

### 2. Load & Explore the Data
```python
loans = pd.read_csv('loan_data.csv')
loans.info()
loans.describe()
loans.head()
```

### 3. Exploratory Data Analysis (EDA)
```python
loans[loans['credit.policy']==1]['fico'].hist(alpha=0.5, color='blue', bins=30, label='Credit Policy=1')
loans[loans['credit.policy']==0]['fico'].hist(alpha=0.5, color='red', bins=30, label='Credit Policy=0')

sns.countplot(x='purpose', hue='not.fully.paid', data=loans, palette='Set1')

sns.jointplot(x='fico', y='int.rate', data=loans, color='purple')

sns.lmplot(x='fico', y='int.rate', data=loans, hue='credit.policy', col='not.fully.paid')
```

### 4. Handle Categorical Features
```python
cat_feats = ['purpose']
final_data = pd.get_dummies(loans, columns=cat_feats, drop_first=True)
```

### 5. Train/Test Split
```python
from sklearn.model_selection import train_test_split

X = final_data.drop('not.fully.paid', axis=1)
y = final_data['not.fully.paid']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
```

### 6. Decision Tree
```python
from sklearn.tree import DecisionTreeClassifier

dtree = DecisionTreeClassifier()
dtree.fit(X_train, y_train)
predictions = dtree.predict(X_test)
```

### 7. Random Forest
```python
from sklearn.ensemble import RandomForestClassifier

rfc = RandomForestClassifier(n_estimators=600)
rfc.fit(X_train, y_train)
rfc_pred = rfc.predict(X_test)
```

### 8. Evaluate Both Models
```python
from sklearn.metrics import classification_report, confusion_matrix

print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))

print(classification_report(y_test, rfc_pred))
print(confusion_matrix(y_test, rfc_pred))
```

---

## Expected Results

### Decision Tree
```
             precision    recall  f1-score   support

          0       0.85      0.81      0.83      2431
          1       0.16      0.20      0.18       443

avg / total       0.74      0.72      0.73      2874

[[1980  451]
 [ 355   88]]
```

### Random Forest (n_estimators=600)
```
             precision    recall  f1-score   support

          0       0.85      1.00      0.92      2431
          1       0.56      0.01      0.02       443

avg / total       0.80      0.85      0.78      2874

[[2427    4]
 [ 438    5]]
```

### Which Performed Better?
**Random Forest** achieved higher overall accuracy (~85% vs ~72%). However, the **Decision Tree** had better recall for class 1 (not fully paid). Random Forest is more reliable for general prediction, while Decision Tree is better at catching risky borrowers.

---

## Requirements

```
pandas
numpy
matplotlib
seaborn
scikit-learn
jupyter
```

Install with:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

---

## How to Run

```bash
jupyter notebook 02-Decision_Trees_and_Random_Forest_Project.ipynb
```
