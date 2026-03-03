# Lab 4: Data Quality Assessment & Preprocessing

## 📌 Project Overview
This repository contains the implementation of **Lab 4** for the **ARTI308 - Machine Learning** course. The project focuses on the essential data preprocessing pipeline using a global car dataset, ensuring the data is clean and optimized for machine learning models.

## 🛠️ Assignment Tasks

### 1. Data Quality Identification
* Conducted a thorough check for missing values and data type consistency using `pandas`.
* Verified that the original `global_cars_enhanced.csv` was clean and ready for transformation.

### 2. Missing Value Management
* **Action:** Artificially introduced missing values (`NaN`) in the `Price_USD` column to demonstrate handling techniques.
* **Strategy:** Applied **Mean Imputation** using `SimpleImputer`.
* **Justification:** Mean imputation is a robust strategy for numerical data that maintains the average value without introducing significant bias.

### 3. Outlier Handling (IQR Method)
* **Target Feature:** `Horsepower`.
* **Method:** Used the **Interquartile Range (IQR)** technique to identify and filter out statistical outliers.
* **Result:** Improved data quality by ensuring extreme values do not negatively impact model training.

### 4. Feature Scaling & Normalization
Implemented and compared two primary normalization techniques:
* **Min-Max Scaling:** Rescaled numerical features into a range of `[0, 1]`.
* **Z-score Normalization (Standardization):** Transformed data to have a mean of `0` and a standard deviation of `1`.

### 5. Principal Component Analysis (PCA)
* **Goal:** Reduce dimensionality while preserving as much variance as possible.
* **Output:** Reduced the feature space to 2 principal components and analyzed the **Explained Variance Ratio** to measure information retention.
---
*Developed as part of the ARTI308 - Machine Learning course.*
