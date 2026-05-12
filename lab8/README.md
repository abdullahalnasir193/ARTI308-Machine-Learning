# K Nearest Neighbors Lab — ARTI308 Machine Learning
 
## Overview
 
This lab walks through building a **K Nearest Neighbors (KNN)** classifier on an anonymous dataset with encrypted feature names. The goal is to predict a target class using scaled features, and find the optimal K value using the Elbow Method.
 
---
 
## Files
 
| File | Description |
|------|-------------|
| `01-K_Nearest_Neighbors.ipynb` | Lecture notebook with examples and explanations |
| `02-K_Nearest_Neighbors_Assignment.ipynb` | Assignment notebook to complete |
| `KNN_Project_Data.csv` | Dataset used for training and testing |
| `Classified_Data.csv` | Additional reference data |
 
---
 
## Dataset — `KNN_Project_Data.csv`
 
The dataset contains anonymized/encrypted feature names with a binary target class.
 
| Column | Description |
|--------|-------------|
| `XVPM` | Encrypted feature |
| `GWYH` | Encrypted feature |
| `TRAT` | Encrypted feature |
| `TLLZ` | Encrypted feature |
| `IGGA` | Encrypted feature |
| `HYKR` | Encrypted feature |
| `EDFS` | Encrypted feature |
| `GUUB` | Encrypted feature |
| `MGJM` | Encrypted feature |
| `JHZC` | Encrypted feature |
| `TARGET CLASS` | **Target variable** — 0 or 1 |
 
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
df = pd.read_csv('KNN_Project_Data.csv')
df.head()
```
 
### 3. Exploratory Data Analysis (EDA)
```python
sns.pairplot(df, hue='TARGET CLASS')
```
 
### 4. Standardize Features
KNN is distance-based, so scaling is critical.
```python
from sklearn.preprocessing import StandardScaler
 
scaler = StandardScaler()
scaler.fit(df.drop('TARGET CLASS', axis=1))
scaled_features = scaler.transform(df.drop('TARGET CLASS', axis=1))
df_feat = pd.DataFrame(scaled_features, columns=df.columns[:-1])
```
 
### 5. Train/Test Split
```python
from sklearn.model_selection import train_test_split
 
X_train, X_test, y_train, y_test = train_test_split(
    df_feat, df['TARGET CLASS'], test_size=0.30, random_state=101)
```
 
### 6. Train KNN with K=1
```python
from sklearn.neighbors import KNeighborsClassifier
 
knn = KNeighborsClassifier(n_neighbors=1)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)
```
 
### 7. Evaluate Initial Model
```python
from sklearn.metrics import classification_report, confusion_matrix
 
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
```
 
### 8. Elbow Method — Find Best K
```python
error_rate = []
 
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))
 
plt.figure(figsize=(10, 6))
plt.plot(range(1, 40), error_rate, color='blue', linestyle='dashed',
         marker='o', markerfacecolor='red', markersize=10)
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')
plt.show()
```
 
### 9. Retrain with Best K
```python
knn = KNeighborsClassifier(n_neighbors=30)
knn.fit(X_train, y_train)
pred = knn.predict(X_test)
 
print(confusion_matrix(y_test, pred))
print(classification_report(y_test, pred))
```
 
---
 
## Expected Results
 
```
WITH K=30
 
[[127  25]
 [ 23 125]]
 
             precision    recall  f1-score   support
 
          0       0.85      0.84      0.84       152
          1       0.83      0.84      0.84       148
 
avg / total       0.84      0.84      0.84       300
```
 
The model achieves **~84% accuracy** with K=30.
 
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
jupyter notebook 02-K_Nearest_Neighbors_Assignment.ipynb
```
