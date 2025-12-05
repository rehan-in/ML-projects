----------------------------------------------

# Home Price Prediction using Linear Regression

This project demonstrates a simple Machine Learning model using Linear Regression to predict home prices based on area (square feet).
It uses Python, Pandas, NumPy, Matplotlib, and Scikit-learn.

----------------------------------------------

# Project Overview

We use a dataset (`homeprices.csv`) containing:

| area (sq ft) | price (USD)  |
| ------------ | ------------ |
| Example data | Example data |

The goal is to build a linear regression model that learns the relationship:

price = m * area + b

Where:

m → slope (coefficient)
b → intercept

Then we use the model to **predict home prices** for new area values.

----------------------------------------------

# Libraries Used

python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model

----------------------------------------------

# Steps Performed

### ✔ 1. Load the data

Read CSV file:

python
df = pd.read_csv("homeprices.csv")
df

### ✔ 2. Visualize the data

python
%matplotlib inline
plt.title("Home Prices")
plt.xlabel("area(sqr ft)")
plt.ylabel("price(US$)")
plt.scatter(df.area, df.price, color='red', marker='+')

### ✔ 3. Train Linear Regression Model

python
reg = linear_model.LinearRegression()
reg.fit(df[['area']], df.price)

### ✔ 4. Make Prediction

Predict price of a **3300 sq ft** home:

python
reg.predict([[3300]])

### ✔ 5. View Model Parameters

python
reg.coef_
reg.intercept_

----------------------------------------------

# Result

The model finds the best-fit line for the data and predicts house prices accurately based on area.

----------------------------------------------

# Requirements

Install required libraries using:

bash
pip install pandas numpy matplotlib scikit-learn

----------------------------------------------

# How to Run This Notebook

1. Open VS Code / Jupyter Notebook
2. Place `homeprices.csv` in the same folder
3. Run all cells in the notebook
4. Check model coefficients & predictions

----------------------------------------------

# Author

Mohammad Rehan
B.tech 3rd Year, ECE branch NIT Mizoram
ML & AI Learner | Python Programmer | VLSI Learner