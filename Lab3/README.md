# Global Car Market Data Analysis (EDA)

## Project Introduction
In this project, I performed an Exploratory Data Analysis (EDA) on a dataset containing information about various car brands and their specifications. The main goal was to understand the relationships between technical features—like horsepower and engine size—and how they impact the market price of the vehicle.

## About the Dataset
The analysis is based on the **Global Cars Enhanced** dataset, which includes 300 unique car records. Each record provides detailed information such as:
* **Vehicle Identity:** Brand, Body Type, and Manufacturing Country.
* **Technical Specs:** Engine CC, Horsepower, Fuel Type, and Transmission.
* **Economic Data:** Price in USD and Mileage (efficiency).
* **Calculated Features:** Car Age and Efficiency Scores.

## Key Steps & Analysis
To ensure the quality of the insights, I followed these steps:
1. **Data Inspection:** Checked for missing values and verified data types using Pandas.
2. **Duplicate Removal:** Used `drop_duplicates()` to ensure every record is unique.
3. **Statistical Profiling:** Generated summaries to understand the average price points and performance metrics.
4. **Data Visualization:** Created several charts using Matplotlib and Seaborn to visualize:
   * The distribution of fuel types (Petrol, Diesel, Hybrid, etc.).
   * The average price per brand to identify premium vs. budget manufacturers.
   * The correlation between Horsepower and Price.

## Tools & Libraries
* **Python:** The core programming language.
* **Pandas:** For data manipulation and cleaning.
* **Seaborn & Matplotlib:** For creating the statistical visualizations.
* **Google Colab:** The environment used for running the analysis.

## Conclusion
This EDA provides a clear picture of how car prices are distributed globally and helps identify which brands offer the most "power for the price." It serves as a foundation for building future machine learning models to predict car prices.
