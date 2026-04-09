# Ecommerce Customers - Regression Analysis 🛒

This repository contains a linear regression project aimed at predicting the **Yearly Amount Spent** by customers of an Ecommerce platform. The project focuses on identifying whether the company should prioritize improvements to their **Mobile App** or their **Website**.

## 📝 Project Tasks
- [x] Load and clean the Ecommerce Customers dataset.
- [x] Perform Exploratory Data Analysis (EDA) using `seaborn`.
- [x] Split data into Training and Testing sets.
- [x] Train a Linear Regression model.
- [x] Evaluate model performance using regression metrics (MAE, MSE, RMSE).
- [x] Analyze coefficients to provide business recommendations.

## 📊 Dataset Features
The dataset includes:
* **Avg. Session Length**: Average session of in-store style advice sessions.
* **Time on App**: Average time spent on App in minutes.
* **Time on Website**: Average time spent on Website in minutes.
* **Length of Membership**: How many years the customer has been a member.
* **Yearly Amount Spent**: (Target Variable) Total annual expenditure per customer.

## 🚀 Key Insights
* **Exploratory Analysis**: A strong correlation was observed between the **Length of Membership** and the **Yearly Amount Spent**.
* **Model Accuracy**: The model was evaluated using RMSE to ensure minimal deviation between predicted and actual spending.
* **Conclusion**: By analyzing the `coef_`, we can determine the specific impact of an extra minute on the App vs. the Website on total revenue.

## 🛠️ Tech Stack
- **Language**: Python
- **Libraries**: Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn

---
*Developed as part of the ARTI308: Machine Learning Course at IAU.*
