# Credit Card Customer Segmentation

## Overview

This project uses **K-Means Clustering** to segment credit card customers based on their usage behavior. It is an unsupervised learning problem — the dataset has no predefined labels, and the goal is to discover hidden patterns to help design better marketing strategies.

**Course:** ARTI308 - Machine Learning

## Dataset

**Source:** [Kaggle - CC General Dataset](https://www.kaggle.com/datasets/arjunbhasin2013/ccdata/data)  
**File:** `CC_GENERAL.csv`  
**Size:** 8,950 customers × 18 features

Each row represents one credit card holder. Key features include:

| Feature | Description |
|---|---|
| `BALANCE` | Outstanding balance on the card |
| `PURCHASES` | Total purchases made |
| `CASH_ADVANCE` | Cash withdrawn using the card |
| `CREDIT_LIMIT` | Credit limit assigned to the customer |
| `PAYMENTS` | Total payments made |
| `MINIMUM_PAYMENTS` | Minimum payments made |
| `TENURE` | Number of months as a customer |

## Project Structure

```
├── 01-Customer_Segmentation_with_K-Means.ipynb     # Lecture notebook
├── 02-Credit_Card_Segmentation_SOLVED.ipynb        # Solved assignment
├── CC_GENERAL.csv                                  # Credit card dataset
├── mall_customers.csv                              # Additional dataset
└── README.md
```

## Requirements

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

## Steps Covered

### 1. Data Loading & Exploration
```python
df = pd.read_csv('CC_GENERAL.csv')
df.head()
df.info()
df.describe()
```

### 2. Data Cleaning
- Dropped `CUST_ID` (identifier, not behavioral)
- Handled missing values using **mean imputation**:
  - `CREDIT_LIMIT` → 1 missing value
  - `MINIMUM_PAYMENTS` → 313 missing values

```python
df.drop('CUST_ID', axis=1, inplace=True)
df.fillna(df.mean(), inplace=True)
```

### 3. Exploratory Data Analysis
- Histograms for all features
- Correlation heatmap
- Scatter plots: `BALANCE` vs `PURCHASES` and `BALANCE` vs `CASH_ADVANCE`

### 4. Feature Scaling
K-Means is distance-based, so scaling is essential:
```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df)
```

### 5. Choosing K
Two methods were used:

**Elbow Method** — inertia drops sharply until K=3, then flattens.

**Silhouette Score:**

| K | Silhouette Score |
|---|---|
| 2 | 0.2100 |
| **3** | **0.2506** ✅ |
| 4 | 0.1976 |
| 5 | 0.1932 |

→ **K = 3** was selected based on both methods.

### 6. Final Model
```python
kmeans_final = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans_final.fit(X_scaled)
df['Cluster'] = kmeans_final.labels_
```

### 7. PCA Visualization
Since the data has 17 features, PCA was used to compress it into 2D for visualization only:
```python
pca = PCA(n_components=2, random_state=42)
pca_components = pca.fit_transform(X_scaled)
```

## Results — Customer Segments

| Cluster | Size | Balance | Purchases | Cash Advance | Credit Limit | Description |
|---|---|---|---|---|---|---|
| **0** | 1,596 | 3,989 | 385 | 3,870 | 6,683 | Cash Advance Users |
| **1** | 1,235 | 2,220 | 4,269 | 458 | 7,734 | High-Value Shoppers |
| **2** | 6,119 | 800 | 506 | 330 | 3,270 | Inactive / Low-Usage |

### Business Interpretation

- **Cluster 0 — Cash Advance Users:** Rely heavily on cash withdrawals. High balance and low purchases. Target with lower interest rate products or financial planning offers.
- **Cluster 1 — High-Value Shoppers:** Most active customers. High purchases, high credit limit, high payments. Target with loyalty rewards, cashback, and premium card upgrades.
- **Cluster 2 — Inactive Customers:** Largest group. Low activity across all metrics. Target with re-engagement campaigns and personalized promotions.

## Key Concepts

| Concept | Description |
|---|---|
| **K-Means** | Assigns each point to the nearest cluster centroid, minimizing inertia |
| **Inertia** | Sum of squared distances from each point to its cluster center |
| **Elbow Method** | Plots inertia vs K to find the point of diminishing returns |
| **Silhouette Score** | Measures how well-separated clusters are (higher = better) |
| **StandardScaler** | Normalizes features to have mean=0 and std=1 |
| **PCA** | Reduces high-dimensional data to 2D for visualization |
