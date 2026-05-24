# Support Vector Machines - Iris Dataset Assignment

## Overview

This project applies Support Vector Machine (SVM) classification on the famous **Iris flower dataset** using Python and scikit-learn. The goal is to classify iris flowers into three species based on their sepal and petal measurements.

## Dataset

The [Iris dataset](http://en.wikipedia.org/wiki/Iris_flower_data_set) contains 150 samples across 3 species:

| Species | Samples |
|---|---|
| Iris Setosa | 50 |
| Iris Versicolor | 50 |
| Iris Virginica | 50 |

**Features:**
- `sepal_length` — Sepal length (cm)
- `sepal_width` — Sepal width (cm)
- `petal_length` — Petal length (cm)
- `petal_width` — Petal width (cm)

## Project Structure

```
├── 01-Support_Vector_Machines.ipynb   # Lecture notebook (breast cancer dataset)
├── 02-SVM_Assignment_SOLVED.ipynb     # Solved assignment (Iris dataset)
└── README.md
```

## Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Steps Covered

### 1. Load Data
```python
iris = sns.load_dataset('iris')
```

### 2. Exploratory Data Analysis
- **Pairplot** — visualize relationships between all features colored by species
- **KDE Plot** — kernel density estimate of sepal dimensions for Setosa only

> 💡 **Finding:** Iris Setosa is the most separable species visually.

### 3. Train/Test Split
```python
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=101)
```

### 4. Train SVM Model
```python
from sklearn.svm import SVC
svc_model = SVC()
svc_model.fit(X_train, y_train)
```

### 5. Model Evaluation
Evaluated using:
- **Confusion Matrix**
- **Classification Report** (Precision, Recall, F1-Score)

### 6. Hyperparameter Tuning with GridSearchCV
```python
param_grid = {
    'C': [0.1, 1, 10, 100],
    'gamma': [1, 0.1, 0.01, 0.001],
    'kernel': ['rbf']
}
grid = GridSearchCV(SVC(), param_grid, verbose=3, cv=5, refit=True)
grid.fit(X_train, y_train)
```

GridSearch automatically finds the best combination of `C` and `gamma` using 5-fold cross-validation.

## Key Concepts

| Concept | Description |
|---|---|
| **SVM** | Finds the optimal hyperplane that maximizes the margin between classes |
| **RBF Kernel** | Maps data to higher dimensions for non-linear separation |
| **C parameter** | Controls the trade-off between a smooth decision boundary and classifying training points correctly |
| **Gamma parameter** | Defines how far the influence of a single training example reaches |
| **GridSearchCV** | Exhaustive search over a parameter grid with cross-validation |

## Course

**ARTI308 - Machine Learning**
