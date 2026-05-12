# Logistic Regression Lab — ARTI308 Machine Learning
 
## Overview
 
This lab walks through building a **Logistic Regression** model to predict whether an internet user will click on an advertisement, based on behavioral and demographic features.
 
---
 
## Files
 
| File | Description |
|------|-------------|
| `01-Logistic_Regression.ipynb` | Lecture notebook with examples and explanations |
| `02-Logistic_Regression_Assignment.ipynb` | Assignment notebook to complete |
| `advertising.csv` | Dataset used for training and testing |
| `titanic_train.csv` | Titanic training data (reference) |
| `titanic_test.csv` | Titanic test data (reference) |
 
---
 
## Dataset — `advertising.csv`
 
The dataset contains 1,000 records of internet users with the following columns:
 
| Column | Description |
|--------|-------------|
| `Daily Time Spent on Site` | Minutes spent on site per day |
| `Age` | Age of the user |
| `Area Income` | Average income in the user's area |
| `Daily Internet Usage` | Average daily minutes on the internet |
| `Ad Topic Line` | Headline of the advertisement |
| `City` | City of the user |
| `Male` | 1 if male, 0 if female |
| `Country` | Country of the user |
| `Timestamp` | Time the user clicked or closed the ad |
| `Clicked on Ad` | **Target variable** — 1 = clicked, 0 = did not click |
 
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
ad_data = pd.read_csv('advertising.csv')
ad_data.head()
ad_data.info()
ad_data.describe()
```
 
### 3. Exploratory Data Analysis (EDA)
- Histogram of `Age`
- Joint plot: `Age` vs `Area Income`
- KDE joint plot: `Age` vs `Daily Time Spent on Site`
- Joint plot: `Daily Time Spent on Site` vs `Daily Internet Usage`
- Pair plot colored by `Clicked on Ad`
### 4. Train/Test Split
```python
from sklearn.model_selection import train_test_split
 
X = ad_data[['Daily Time Spent on Site', 'Age', 'Area Income',
             'Daily Internet Usage', 'Male']]
y = ad_data['Clicked on Ad']
 
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
```
 
### 5. Train the Model
```python
from sklearn.linear_model import LogisticRegression
 
logmodel = LogisticRegression()
logmodel.fit(X_train, y_train)
```
 
### 6. Evaluate
```python
from sklearn.metrics import classification_report, confusion_matrix
 
predictions = logmodel.predict(X_test)
print(classification_report(y_test, predictions))
print(confusion_matrix(y_test, predictions))
```
 
---
 
## Expected Results
 
```
              precision    recall  f1-score   support
 
           0       0.86      0.96      0.91       162
           1       0.96      0.85      0.90       168
 
    accuracy                           0.91       330
```
 
The model achieves **~91% accuracy** on the test set.
 
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
jupyter notebook 02-Logistic_Regression_Assignment.ipynb
```
